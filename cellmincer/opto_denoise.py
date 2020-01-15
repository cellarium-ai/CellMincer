import numpy as np
from skimage.filters import threshold_otsu
import torch
import logging
from typing import List, Tuple, Optional, Union, Dict

from .opto_ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace
from .opto_utils import pad_images_torch, crop_center, get_nn_spatio_temporal_mean

logger = logging.getLogger()

    
class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, unet_kernel_size, activation):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(torch.nn.Conv2d(in_size, out_size, kernel_size=unet_kernel_size, padding=int(padding)))
        block.append(activation)
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))

        block.append(torch.nn.Conv2d(out_size, out_size, kernel_size=unet_kernel_size, padding=int(padding)))
        block.append(activation)
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(torch.nn.Module):
    def __init__(self,
                 in_size: int,
                 mid_size: int,
                 bridge_size: int,
                 out_size: int,
                 up_mode: 'str',
                 padding: int,
                 batch_norm: bool,
                 unet_kernel_size: int,
                 activation: torch.nn.Module):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = torch.nn.ConvTranspose2d(in_size, mid_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(mode='bilinear', scale_factor=2),
                torch.nn.Conv2d(in_size, mid_size, kernel_size=1))
        
        self.conv_block = UNetConvBlock(
            mid_size + bridge_size, out_size, padding, batch_norm, unet_kernel_size, activation)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2

        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)  # cat along channels
        out = self.conv_block(out)
        return out

    
class UNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int,
            wf: int,
            out_channels_before_readout: int,
            padding: bool = False,
            batch_norm: bool = False,
            skip_readout: bool = False,
            up_mode: str = 'upconv',
            unet_kernel_size: int = 3,
            readout_hidden_layer_channels_list: List[int] = [],
            readout_kernel_size: int = 1,
            activation: torch.nn.Module = torch.nn.SELU(),
            device: torch.device = torch.device('cuda'),
            dtype: torch.dtype = torch.float32):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wf = wf
        self.depth = depth
        
        self.device = device
        self.dtype = dtype
        
        # downward path
        prev_channels = in_channels
        self.down_path = torch.nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(
                    prev_channels,
                    2 ** (wf + i),
                    padding,
                    batch_norm,
                    unet_kernel_size,
                    activation))
            prev_channels = 2 ** (wf + i)

        # upward path
        self.up_path = torch.nn.ModuleList()
        for i in reversed(range(depth - 1)):
            if i > 0:
                up_in_channels = prev_channels
                up_bridge_channels = 2 ** (wf + i)
                up_mid_channels = 2 ** (wf + i)
                up_out_channels = 2 ** (wf + i)
            else:
                up_in_channels = prev_channels
                up_bridge_channels = 2 ** (wf + i)
                up_mid_channels = 2 ** (wf + i)
                up_out_channels = out_channels_before_readout
            self.up_path.append(
                UNetUpBlock(
                    up_in_channels,
                    up_mid_channels,
                    up_bridge_channels,
                    up_out_channels,
                    up_mode,
                    padding,
                    batch_norm,
                    unet_kernel_size,
                    activation))
            prev_channels = up_out_channels

        # final readout
        if not skip_readout:
            readout = []
            for hidden_channels in readout_hidden_layer_channels_list:
                readout.append(
                    torch.nn.Conv2d(
                        prev_channels,
                        hidden_channels,
                        kernel_size=readout_kernel_size))
                readout.append(activation)
                prev_channels = hidden_channels
            readout.append(
                torch.nn.Conv2d(
                    prev_channels,
                    out_channels,
                    kernel_size=readout_kernel_size))
            self.readout = torch.nn.Sequential(*readout)
        else:
            self.readout = None
        
        # send to device
        self.to(device)
        
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.functional.F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            
        if self.readout is not None:
            return self.readout(x)
        else:
            return x
    

def get_unet_input_size(output_min_size: int, kernel_size: int, depth: int):
    """Smallest input size for output size >= `output_min_size`.
    
    .. note:
        The calculated input sizes guarantee that all of the layers have even dimensions.
        This is important to prevent aliasing in downsampling (pooling) operations.
    
    """
    delta = 2 * (kernel_size - 1)
    pad = delta * sum([2 ** i for i in range(depth)])
    ds = 2 ** depth
    res = (output_min_size + pad) % ds
    bottom_size = (output_min_size + pad) // ds + res
    input_size = bottom_size
    for i in range(depth):
        input_size = 2 * (input_size + delta)
    input_size += delta
    return input_size


def get_minimum_padding(
        x_window: int,
        y_window: int,
        denoiser_config: Dict[str, Union[int, float, str]]) -> Tuple[int, int]:
    
    padded_x_window = get_unet_input_size(
        output_min_size=(
            x_window
            + (denoiser_config['unet_readout_kernel_size'] - 1)
                * len(denoiser_config['unet_readout_hidden_layer_channels_list']) + 1),
        kernel_size=denoiser_config['unet_kernel_size'],
        depth=denoiser_config['unet_depth'])
    padded_y_window = get_unet_input_size(
        output_min_size=(
            y_window
            + (denoiser_config['unet_readout_kernel_size'] - 1)
                * len(denoiser_config['unet_readout_hidden_layer_channels_list']) + 1),
        kernel_size=denoiser_config['unet_kernel_size'],
        depth=denoiser_config['unet_depth'])

    x_padding = (padded_x_window - x_window) // 2
    y_padding = (padded_y_window - y_window) // 2
    
    return x_padding, y_padding

    
def generate_bernoulli_mask(
        p: float,
        n_batch: int,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    return torch.distributions.Bernoulli(
        probs=torch.tensor(p, device=device, dtype=dtype)).sample(
        [n_batch, width, height]).type(dtype)


def generate_bernoulli_mask_on_mask(
        p: float,
        in_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    out_mask = torch.zeros(in_mask.shape, device=device, dtype=dtype)
    active_pixels = in_mask.sum()
    bern = torch.distributions.Bernoulli(
        probs=torch.tensor(p, device=device, dtype=dtype)).sample(
        [active_pixels.item()]).type(dtype)
    out_mask[in_mask] = bern
    return out_mask
    
    
def generate_occluded_training_data(
        ws_base_list: List[OptopatchBaseWorkspace],
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        t_order: int,
        t_tandem: int,
        n_batch: int,
        x_window: int,
        y_window: int,
        occlusion_prob: float,
        occlusion_strategy: str,
        continuity_reg_strength: float,
        noise_threshold_to_std: float,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
    """Generates minibatches with appropriate occlusion and padding for training a blind
    denoiser. Supports multiple datasets.
    
    The temporal length of each slice is:
    
        | (t_order - 1) // 2 | t_tandem + 1 | (t_order - 1) // 2 | 
    
    If `t_tandem == 0`, then only a single frame in the middle frame is Bernoulli-occluded and
    expected to be used for Noise2Self training. If `t_tandem > 0`, then a total number of
    `t_tandem + 1` frames sandwiched in the middle will be randomly Bernoulli-occluded.
    
    """
    
    assert t_order % 2 == 1
    assert t_tandem % 2 == 0
    assert occlusion_strategy in {'random', 'nn-average'}
    
    padded_x_window = x_window + 2 * ws_denoising_list[0].x_padding
    padded_y_window = y_window + 2 * ws_denoising_list[0].y_padding
    
    n_datasets = len(ws_denoising_list)
    n_global_features = ws_denoising_list[0].n_global_features
    mean_f_index = ws_denoising_list[0].feature_names.index('mean')
    std_f_index = ws_denoising_list[0].feature_names.index('std')
    global_features_f_begin_index = 0
    global_features_f_end_index = n_global_features
    movie_f_index = n_global_features
    n_input_features = n_global_features + 1
    
    t_total = t_order + t_tandem
    t_mid = (t_order + t_tandem - 1) // 2
    
    # sample random dataset indices
    dataset_indices = np.random.randint(0, n_datasets, size=n_batch)

    # random time slices
    time_slice_locs = np.random.rand(n_batch)
    t_begin_indices = [
        int(np.floor((ws_base_list[i_dataset].n_frames - t_total) * loc))
        for i_dataset, loc in zip(dataset_indices, time_slice_locs)]
    t_end_indices = [
        int(np.floor((ws_base_list[i_dataset].n_frames - t_total) * loc)) + t_total
        for i_dataset, loc in zip(dataset_indices, time_slice_locs)]
    
    # random space slices
    x0_list = [
        np.random.randint(0, ws_base_list[i_dataset].width - x_window + 1)
        for i_dataset in dataset_indices]
    y0_list = [
        np.random.randint(0, ws_base_list[i_dataset].height - y_window + 1)
        for i_dataset in dataset_indices]
        
    # generate a uniform bernoulli mask
    n_total_masks = n_batch * (t_tandem + 1)
    occlusion_masks_ntxy = generate_bernoulli_mask(
        p=occlusion_prob,
        n_batch=n_total_masks,
        width=x_window,
        height=y_window,
        device=device,
        dtype=dtype).view(n_batch, t_tandem + 1, x_window, y_window)    
    
    # slice the movies and copy to device
    random_slice_data_list = [
        ws_denoising_list[dataset_indices[i_batch]].get_slice(
            t_begin_index=t_begin_indices[i_batch],
            t_end_index=t_end_indices[i_batch],
            x0=x0_list[i_batch],
            y0=y0_list[i_batch],
            x_window=x_window,
            y_window=y_window)
        for i_batch in range(n_batch)]
    
    padded_sliced_movie_ntxy = torch.tensor(
        np.concatenate(
            tuple(random_slice_data_list[i_batch][0] for i_batch in range(n_batch)),
            axis=0), device=device, dtype=dtype)
    
    padded_global_features_nfxy = torch.tensor(
        np.concatenate(
            tuple(random_slice_data_list[i_batch][1] for i_batch in range(n_batch)),
            axis=0), device=device, dtype=dtype)

    # make a hard copy of the to-be-occluded frames
    padded_middle_frames_ntxy = padded_sliced_movie_ntxy[
        :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), ...].clone()
    
    # pad the mask to match the padded movie
    padded_occlusion_masks_ntxy = pad_images_torch(
        images_ncxy=occlusion_masks_ntxy,
        target_width=padded_x_window,
        target_height=padded_y_window,
        pad_value_nc=torch.zeros(n_batch, t_tandem + 1, device=device, dtype=dtype))

    if occlusion_strategy == 'nn-average':
        
        padded_sliced_movie_ntxy[
            :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] *= (
                1 - padded_occlusion_masks_ntxy)
        
        for i_t in range(t_tandem + 1):
            padded_sliced_movie_ntxy[:, t_mid - (t_tandem // 2) + i_t, 1:-1, 1:-1] += (
                padded_occlusion_masks_ntxy[:, i_t, 1:-1, 1:-1]
                * get_nn_spatio_temporal_mean(
                    padded_sliced_movie_ntxy, t_mid - (t_tandem // 2) + i_t))
    
    elif occlusion_strategy == 'random':

        padded_sliced_movie_ntxy[
            :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] = (
                (1 - padded_occlusion_masks_ntxy) * padded_sliced_movie_ntxy[
                    :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :]
                + padded_occlusion_masks_ntxy * torch.distributions.Normal(
                    loc=padded_global_features_nfxy[:, mean_f_index, :, :][:, None, ...].expand(
                        n_batch, t_tandem + 1, padded_x_window, padded_y_window),
                    scale=padded_global_features_nfxy[:, std_f_index, :, :][:, None, ...].expand(
                        n_batch, t_tandem + 1, padded_x_window, padded_y_window)).sample())
    
    else:
            
        raise ValueError("Unknown occlusion strategy; valid options: 'nn-average', 'random'")
                
    return {
        'padded_global_features_nfxy': padded_global_features_nfxy,  # global constant feature
        'padded_sliced_movie_ntxy': padded_sliced_movie_ntxy,  # sliced movie with occluded pixels 
        'padded_middle_frames_ntxy': padded_middle_frames_ntxy,  # original frames in the middle of the movie
        'padded_occlusion_masks_ntxy': padded_occlusion_masks_ntxy,  # occlusion masks
        'x_window': x_window,
        'y_window': y_window,
        'padded_x_window': padded_x_window,
        'padded_y_window': padded_y_window,
        'mean_f_index': mean_f_index,
        'std_f_index': std_f_index,
        'continuity_reg_strength': continuity_reg_strength,
        'noise_threshold_to_std': noise_threshold_to_std
    }


def get_loss_end_to_end(
        batch_data,
        ws_base_list: List[OptopatchBaseWorkspace],
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        end_to_end_denoiser: torch.nn.Module,
        norm_p: int,
        enable_continuity_reg: bool,
        reg_func: str,
        eps: float = 1e-6):
    """Calculates the loss of a blind U-Net denoiser on a given minibatch."""
    
    assert reg_func in {'clamped_linear', 'tanh'}
    
    device = batch_data['padded_sliced_movie_ntxy'].device
    dtype = batch_data['padded_sliced_movie_ntxy'].dtype
    x_window = batch_data['x_window']
    y_window = batch_data['y_window']
    padded_x_window = batch_data['padded_x_window']
    padded_y_window = batch_data['padded_y_window']
    n_batch, t_total = batch_data['padded_sliced_movie_ntxy'].shape[:2]
    n_global_features = batch_data['padded_global_features_nfxy'].shape[-3]
    n_input_features = n_global_features + 1
    movie_f_index = n_global_features
    t_tandem = batch_data['padded_occlusion_masks_ntxy'].shape[-3] - 1
    t_order = t_total - t_tandem
    t_mid = (t_order + t_tandem - 1) // 2

    # iterate over the middle frames and accumulate loss
    def add_to_loss(_loss, _err, _norm_p=norm_p, _scale=1.):
        _new_loss = (_scale * _err.abs().pow(_norm_p)).sum()
        if _loss is None:
            return _new_loss
        else:
            return _loss + _new_loss
    
    cropped_movie_t_std_nxy = crop_center(
        batch_data['padded_global_features_nfxy'][:, batch_data['std_f_index'], ...],
        target_width=x_window,
        target_height=y_window)

    rec_loss = None
    reg_loss = None
    prev_cropped_denoised_output_nxy = None

    # calcualate the loss for every frame; also, TV loss between frames if enabled
    for i_t in range(t_tandem + 1):

        denoiser_input_ncxy = torch.cat(
            (batch_data['padded_global_features_nfxy'],
             batch_data['padded_sliced_movie_ntxy'][:, i_t:(i_t + t_order), ...]),
            dim=-3)
        
        # combine
        denoiser_output_nxy = end_to_end_denoiser(denoiser_input_ncxy)[:, 0, ...]

        # crop mask
        cropped_mask_nxy = crop_center(
            batch_data['padded_occlusion_masks_ntxy'][:, i_t, ...],
            target_width=x_window,
            target_height=y_window)
        
        # crop denoiser output
        cropped_denoiser_output_nxy = crop_center(
            denoiser_output_nxy,
            target_width=x_window,
            target_height=y_window)
        
        # crop expected output
        expected_output_nxy = crop_center(
            batch_data['padded_middle_frames_ntxy'][:, i_t, ...],
            target_width=x_window,
            target_height=y_window)

        # reconstruction loss
        err = cropped_mask_nxy * (cropped_denoiser_output_nxy - expected_output_nxy)
        rec_loss = add_to_loss(rec_loss, err)
                
        # temporal continuity loss
        if enable_continuity_reg:
            
            if i_t > 0:
                
                # need to scale the regularization by masked fraction to make
                # the scaling of reconstruction loss and regularization commensurate
                # with each other
                
                total_masked_pixels_n = cropped_mask_nxy.sum((-1, -2)).type(dtype)
                total_pixels = x_window * y_window
                masked_fraction_n = total_masked_pixels_n / total_pixels
                
                if reg_func == 'clamped_linear':
                    
                    soft_thresholded_delta_nxy = torch.clamp(
                        (cropped_denoiser_output_nxy - prev_cropped_denoiser_output_nxy) / (
                            eps + cropped_movie_t_std_nxy),
                        min=0.,
                        max=batch_data['noise_threshold_to_std'])
                    
                elif reg_func == 'tanh':
                    
                    eta = eps + batch_data['noise_threshold_to_std']
                    soft_thresholded_delta_nxy = eta * torch.tanh(
                        (cropped_denoiser_output_nxy - prev_cropped_denoiser_output_nxy) / (
                            (eps + cropped_movie_t_std_nxy) * eta))
                    
                reg_loss = add_to_loss(
                    reg_loss, soft_thresholded_delta_nxy, norm_p,
                    batch_data['continuity_reg_strength'] * masked_fraction_n[:, None, None])
            
            else:
                
                prev_cropped_denoiser_output_nxy = cropped_denoiser_output_nxy

    return rec_loss, reg_loss


def generate_input_for_single_frame_denoising(
        ws_base_list: List[OptopatchBaseWorkspace],
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        i_dataset: int,
        i_t: int,
        t_order: int,
        x0: int,
        y0: int,
        x_window: int,
        y_window: int,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
    """Prepares data for single-dataset single-frame denoising"""
    
    # slice the movies and copy to device
    t_mid = (t_order - 1) // 2
    padded_sliced_movie_1txy, padded_global_features_1fxy = ws_denoising_list[i_dataset].get_slice(
        t_begin_index=i_t - t_mid,
        t_end_index=i_t + t_mid + 1,
        x0=x0,
        y0=y0,
        x_window=x_window,
        y_window=y_window)
    
    padded_sliced_movie_txy = torch.tensor(
        padded_sliced_movie_1txy[0, ...],
        device=device, dtype=dtype)
    
    # constant global features
    padded_global_features_fxy = torch.tensor(
        padded_global_features_1fxy[0, ...],
        device=device, dtype=dtype)
    
    return {
        'padded_global_features_fxy': padded_global_features_fxy,
        'padded_sliced_movie_txy': padded_sliced_movie_txy,
        'x_window': x_window,
        'y_window': y_window
    }


def denoise_end_to_end(batch_data,
                       ws_base_list: List[OptopatchBaseWorkspace],
                       ws_denoising_list: List[OptopatchDenoisingWorkspace],
                       end_to_end_denoiser: torch.nn.Module):
    """Denoises a single frame from a single dataset"""
    
    device = batch_data['padded_sliced_movie_txy'].device
    dtype = batch_data['padded_sliced_movie_txy'].dtype
    t_order, padded_width, padded_height = batch_data['padded_sliced_movie_txy'].shape
    n_global_features = batch_data['padded_global_features_fxy'].shape[-3]
    
    # set to evaluation mode
    end_to_end_denoiser.eval()
    
    with torch.no_grad():
        
        # generate input for per-frame feature extractor
        input_fxy = torch.zeros(
            n_global_features + t_order, padded_width, padded_height,
            device=device, dtype=dtype)

        # copy global features
        input_fxy[:n_global_features, :, :] = batch_data['padded_global_features_fxy']

        # copy movie
        input_fxy[n_global_features:, :, :] = batch_data['padded_sliced_movie_txy']
        
        # combine
        cropped_denoised_output_xy = crop_center(
            end_to_end_denoiser(input_fxy[None, ...]),
            target_width=batch_data['x_window'],
            target_height=batch_data['y_window'])[0, 0, ...]
        
        return cropped_denoised_output_xy

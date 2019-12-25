import numpy as np
from skimage.filters import threshold_otsu
import torch
import logging
from typing import List, Tuple, Optional

from .opto_ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace
from .opto_utils import pad_images_torch, crop_center, get_nn_spatio_temporal_mean

logger = logging.getLogger()

    
class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(torch.nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(torch.nn.SELU())
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))

        block.append(torch.nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(torch.nn.SELU())
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(mode='bilinear', scale_factor=2),
                torch.nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

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
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

    
class UNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            depth: int,
            wf: int,
            padding: bool = False,
            batch_norm: bool = False,
            up_mode: str = 'upconv',
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
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        # upward path
        self.up_path = torch.nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        # final readout
        self.readout = torch.nn.Conv2d(prev_channels, out_channels, kernel_size=1)
        
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

        return self.readout(x)


class ConvolutionalTemporalCombiner(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 kernel_size: int,
                 batch_norm: bool,
                 hidden_layer_channels_list: List[int] = [],
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32):
        super(ConvolutionalTemporalCombiner, self).__init__()
        self.in_channels = int
        self.kernel_size = kernel_size
        
        block = []
        prev_channels = input_channels        
        for hidden_layer_channels in hidden_layer_channels_list:
            block.append(torch.nn.SELU())
            if batch_norm:
                block.append(torch.nn.BatchNorm2d(prev_channels))
            block.append(torch.nn.Conv2d(prev_channels, hidden_layer_channels, kernel_size=kernel_size))
            prev_channels = hidden_layer_channels
        
        block.append(torch.nn.SELU())
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(prev_channels))
        block.append(torch.nn.Conv2d(prev_channels, 1, kernel_size=kernel_size))
        
        self.block = torch.nn.Sequential(*block)
        
        self.to(device)
        
    def forward(self, x):
        out = self.block(x)
        return out


class ConvolutionalTemporalCombiner3D(torch.nn.Module):
    def __init__(self,
                 kernel_size: int,
                 hidden_layer_channels_list: List[int] = [],
                 device: torch.device = torch.device('cuda'),
                 dtype: torch.dtype = torch.float32):
        super(ConvolutionalTemporalCombiner3D, self).__init__()
        self.in_channels = int
        self.kernel_size = kernel_size
        
        block = []
        input_channels = 1
        prev_channels = input_channels        
        for hidden_layer_channels in hidden_layer_channels_list:
            block.append(torch.nn.SELU())
            block.append(torch.nn.Conv3d(prev_channels, hidden_layer_channels, kernel_size=kernel_size))
            prev_channels = hidden_layer_channels
        
        block.append(torch.nn.SELU())
        block.append(torch.nn.Conv3d(prev_channels, 1, kernel_size=kernel_size))
        
        self.block = torch.nn.Sequential(*block)
        
        self.to(device)
        
    def forward(self, x):
        # add a dummy dim for input_channel
        y = x[:, None, ...]
        out = self.block(y)
        
        # assert that we finally have temporal dim 1
        assert out.shape[-3] == 1
        return out[:, :, 0, ...]
    

def get_unet_input_size(output_min_size: int, kernel_size: int, depth: int):
    """Smallest input size for output size >= `output_min_size`"""
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
        movie_width: int,
        movie_height: int,
        padded_width: int,
        padded_height: int,
        occlusion_prob: float,
        only_fg_pixels: bool,
        include_mask: bool = False,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
    """Generates minibatches with appropriate occlusion and padding for training a blind
    denoiser. Supports multiple datasets"""
    
    assert t_order % 2 == 1
    assert t_tandem % 2 == 0
    
    n_datasets = len(ws_denoising_list)
    n_global_features = ws_denoising_list[0].features_1fxy.shape[-3]
    global_features_f_begin_index = 0
    global_features_f_end_index = n_global_features
    t_total = t_order + t_tandem
    t_mid = (t_order + t_tandem - 1) // 2

    if include_mask:
        n_input_features_per_frame = n_global_features + 2
        mask_f_index = n_global_features 
        movie_f_index = n_global_features + 1
    else:
        n_input_features_per_frame = n_global_features + 1
        movie_f_index = n_global_features 
    
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

    if only_fg_pixels:
        
        # fg masks
        in_mask_ntxy = torch.tensor(
            np.concatenate(
                [ws_base_list[i_dataset].cosine_fg_sim_otsu_fg_pixel_mask_xy[None, ...]
                 for i_dataset in dataset_indices], axis=0)[:, None, :, :],
            device=device, dtype=torch.bool).expand(
                [n_batch, t_tandem + 1, movie_width, movie_height])

        # occlusion masks on fg masks
        occlusion_masks_ntxy = generate_bernoulli_mask_on_mask(
            p=occlusion_prob,
            in_mask=in_mask_ntxy,
            device=device,
            dtype=dtype)
        
    else:
        
        # generate a uniform bernoulli mask
        n_total_masks = n_batch * (t_tandem + 1)
        occlusion_masks_ntxy = generate_bernoulli_mask(
            p=occlusion_prob,
            n_batch=n_total_masks,
            width=movie_width,
            height=movie_height,
            device=device,
            dtype=dtype).view(n_batch, t_tandem + 1, movie_width, movie_height)    
    
    # slice the movies and copy to device
    padded_sliced_movie_ntxy = torch.tensor(
        np.concatenate(
            tuple(
                ws_denoising_list[dataset_indices[i_batch]].get_slice(
                    t_begin_indices[i_batch],
                    t_end_indices[i_batch])
                for i_batch in range(n_batch)), axis=0),
        device=device, dtype=dtype)
    
    # make a copy of the to-be-occluded frames
    padded_middle_frames_ntxy = padded_sliced_movie_ntxy[
        :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), ...].clone()
    
    # pad the mask to match the padded movie
    padded_occlusion_masks_ntxy = pad_images_torch(
        images_ncxy=occlusion_masks_ntxy,
        target_width=padded_width,
        target_height=padded_height,
        pad_value_nc=torch.zeros(occlusion_masks_ntxy.shape[:2], device=device, dtype=dtype))

    # occlude the movie with mask
    padded_sliced_movie_ntxy[
        :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] = \
        padded_sliced_movie_ntxy[
            :, (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1), :, :] * (
                1. - padded_occlusion_masks_ntxy)

    # replace masked pixels by spatio-temporal kNN average
    for i_t in range(t_tandem + 1):
        padded_sliced_movie_ntxy[:, t_mid - (t_tandem // 2) + i_t, 1:-1, 1:-1] += (
            padded_occlusion_masks_ntxy[:, i_t, 1:-1, 1:-1]
            * get_nn_spatio_temporal_mean(
                padded_sliced_movie_ntxy, t_mid - (t_tandem // 2) + i_t))

    # put togeher
    feature_enriched_occluded_input_ntfxy = torch.zeros(
        (n_batch, t_total, n_input_features_per_frame, padded_width, padded_height),
        device=device, dtype=dtype)

    # copy global features
    feature_enriched_occluded_input_ntfxy[
        :, :, global_features_f_begin_index:global_features_f_end_index, :, :] = torch.cat(
        tuple(torch.tensor(ws_denoising_list[i_dataset].features_1fxy,
                           device=device, dtype=dtype).unsqueeze(0)
              for i_dataset in dataset_indices), dim=0).expand(
        n_batch, t_total, n_global_features, padded_width, padded_height)

    if include_mask:
        # copy mask
        feature_enriched_occluded_input_ntfxy[
            :,
            (t_mid - (t_tandem // 2)):(t_mid + (t_tandem // 2) + 1),
            mask_f_index:(mask_f_index + 1),
            :,
            :] = padded_occlusion_masks_ntxy

    # copy occluded movie
    feature_enriched_occluded_input_ntfxy[
        :, :, movie_f_index:(movie_f_index + 1), :, :] = padded_sliced_movie_ntxy.view(
        n_batch, t_total, 1, padded_width, padded_height)
                
    return {
        'feature_enriched_occluded_input_ntfxy': feature_enriched_occluded_input_ntfxy,
        'padded_middle_frames_ntxy': padded_middle_frames_ntxy,
        'padded_occlusion_masks_ntxy': padded_occlusion_masks_ntxy,
        'dataset_indices': dataset_indices,
        't_begin_indices': t_begin_indices,
        't_end_indices': t_end_indices
    }


def get_loss(batch_data,
             ws_denoising_list: List[OptopatchDenoisingWorkspace],
             per_frame_feature_extractor: torch.nn.Module,
             temporal_combiner: torch.nn.Module,
             movie_width: int,
             movie_height: int,
             norm_p: int,
             enable_continuity_reg: bool,
             pass_global_features_to_combiner: bool,
             continuity_reg_strength: float):
    """Calculates the loss of a blind denoiser on a given minibatch."""
    # shorthand
    input_ntfxy = batch_data['feature_enriched_occluded_input_ntfxy']
    
    # shapes
    n_batch, t_total, n_input_features, width, height = input_ntfxy.shape
    t_tandem = batch_data['padded_occlusion_masks_ntxy'].shape[-3] - 1
    t_order = t_total - t_tandem
    t_mid = (t_order + t_tandem - 1) // 2
    
    # flatten across batch index and time into one 
    flat_input_features_mfxy = input_ntfxy.view(
        n_batch * t_total, n_input_features, width, height)
    
    # process each batch and frame
    frame_features_mlxy = per_frame_feature_extractor(flat_input_features_mfxy)
    
    # generate input for the combiner
    n_global_features = ws_denoising_list[0].features_1fxy.shape[-3]
    global_features_f_begin_index = 0
    global_features_f_end_index = n_global_features
    global_features_nfxy = crop_center(
        input_ntfxy[:, 0, global_features_f_begin_index:global_features_f_end_index, ...],
        target_width=frame_features_mlxy.shape[-2],
        target_height=frame_features_mlxy.shape[-1])
    
    t_stride = per_frame_feature_extractor.out_channels
    assert t_stride == 1 # let's be conservative for now -- talk to Luca
    frame_features_nqxy = frame_features_mlxy.view(
        (n_batch,
         t_total * t_stride,
         frame_features_mlxy.shape[-2],
         frame_features_mlxy.shape[-1]))
    
    # iterate over the middle frames and accumulate loss
    def add_to_loss(loss, err, norm_p=norm_p, scale=1.):
        if norm_p == 1:
            c_loss = scale * err.abs().sum()
        elif norm_p == 2:
            c_loss = scale * err.pow(2).sum()
        else:
            raise ValueError('Only norm 1 and norm 2 are supported!')
        if loss is None:
            return c_loss
        else:
            return loss + c_loss

    loss = None
    prev_cropped_combiner_output_nxy = None
    
    for i_t in range(t_tandem + 1):

        if pass_global_features_to_combiner:
            combiner_input_ncxy = torch.cat(
                (frame_features_nqxy[:, (i_t * t_stride):((i_t + t_order) * t_stride), ...],
                 global_features_nfxy), dim=-3)
        else:
            combiner_input_ncxy = frame_features_nqxy[:, (i_t * t_stride):((i_t + t_order) * t_stride), ...]
        
        # combine!
        combiner_output_nxy = temporal_combiner(combiner_input_ncxy)[:, 0, ...]

        # crop mask and combiner output
        cropped_mask_nxy = crop_center(
            batch_data['padded_occlusion_masks_ntxy'][:, i_t, ...],
            target_width=movie_width,
            target_height=movie_height)
        cropped_combiner_output_nxy = crop_center(
            combiner_output_nxy,
            target_width=movie_width,
            target_height=movie_height)
        expected_output_nxy = crop_center(
            batch_data['padded_middle_frames_ntxy'][:, i_t, ...],
            target_width=movie_width,
            target_height=movie_height)

        # reconstruction loss
        err = cropped_mask_nxy * (cropped_combiner_output_nxy - expected_output_nxy)
        loss = add_to_loss(loss, err)
        
        # temporal continuity loss
        if enable_continuity_reg:
            if prev_cropped_combiner_output_nxy is not None:
                loss = add_to_loss(
                    err=(cropped_combiner_output_nxy - prev_cropped_combiner_output_nxy),
                    loss=loss,
                    norm_p=norm_p,
                    scale=continuity_reg_strength)
            prev_cropped_combiner_output_nxy = cropped_combiner_output_nxy
        
    return loss


def generate_input_for_denoising(
        i_dataset: int,
        i_t: int,
        n_samp: int,
        ws_base_list: List[OptopatchBaseWorkspace],
        ws_denoising_list: List[OptopatchDenoisingWorkspace],
        t_order: int,
        movie_width: int,
        movie_height: int,
        padded_width: int,
        padded_height: int,
        occlusion_prob: float,
        include_mask: bool = False,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32):
    """Prepares data for single-dataset single-frame denoising"""
    
    n_global_features = ws_denoising_list[0].features_1fxy.shape[-3]
    global_features_f_begin_index = 0
    global_features_f_end_index = n_global_features
    n_batch = n_samp
    
    if include_mask:
        n_input_features_per_frame = n_global_features + 2
        mask_f_index = n_global_features 
        movie_f_index = n_global_features + 1
    else:
        n_input_features_per_frame = n_global_features + 1
        movie_f_index = n_global_features 

    # generate occlusion masks (1 means occlud)
    occlusion_masks_nxy = generate_bernoulli_mask(
        p=occlusion_prob,
        n_batch=n_batch,
        width=movie_width,
        height=movie_height,
        device=device,
        dtype=dtype)

    # slice the movies and copy to device
    t_mid = (t_order - 1) // 2
    padded_sliced_movie_ntxy = torch.tensor(
        np.concatenate(
            tuple(
                ws_denoising_list[i_dataset].get_slice(i_t - t_mid, i_t + t_mid + 1)
                for i_batch in range(n_batch)), axis=0),
        device=device, dtype=dtype)
    
    # make a copy of the middle frame
    padded_middle_frames_nxy = padded_sliced_movie_ntxy[:, t_mid, ...].clone()
    
    # pad the mask to match the padded movie
    padded_occlusion_masks_n1xy = pad_images_torch(
        images_ncxy=occlusion_masks_nxy.view(
            occlusion_masks_nxy.shape[0], 1, movie_width, movie_height),
        target_width=padded_width,
        target_height=padded_height,
        pad_value_nc=torch.zeros(
            (occlusion_masks_nxy.shape[0], 1), device=device, dtype=dtype))
    
    # occlude the movie with mask
    padded_sliced_movie_ntxy[:, t_mid, :, :] = padded_sliced_movie_ntxy[:, t_mid, :, :] * (
        1. - padded_occlusion_masks_n1xy[:, 0, ...])

    # replace masked pixels by spatio-temporal kNN average
    padded_sliced_movie_ntxy[:, t_mid, 1:-1, 1:-1] += (
        padded_occlusion_masks_n1xy[:, 0, 1:-1, 1:-1]
        * get_nn_spatio_temporal_mean(padded_sliced_movie_ntxy, t_mid))

    # put togeher
    feature_enriched_occluded_input_ntfxy = torch.zeros(
        (n_batch, t_order, n_input_features_per_frame, padded_width, padded_height),
        device=device, dtype=dtype)

    # copy global features
    feature_enriched_occluded_input_ntfxy[
        :, :, global_features_f_begin_index:global_features_f_end_index, :, :] = torch.cat(
        tuple(torch.tensor(ws_denoising_list[i_dataset].features_1fxy,
                           device=device, dtype=dtype).unsqueeze(0)
              for i_batch in range(n_batch)), dim=0).expand(
        n_batch, t_order, n_global_features, padded_width, padded_height)

    if include_mask:
        # copy mask
        feature_enriched_occluded_input_ntfxy[
            :, t_mid, mask_f_index:(mask_f_index + 1), :, :] = padded_occlusion_masks_n1xy

    # copy occluded movie
    feature_enriched_occluded_input_ntfxy[
        :, :, movie_f_index:(movie_f_index + 1), :, :] = padded_sliced_movie_ntxy.view(
        n_batch, t_order, 1, padded_width, padded_height)
    
    return {
        'feature_enriched_occluded_input_ntfxy': feature_enriched_occluded_input_ntfxy}


def denoise(denoising_input_data,
            ws_denoising_list: List[OptopatchDenoisingWorkspace],
            per_frame_feature_extractor: torch.nn.Module,
            temporal_combiner: torch.nn.Module,
            pass_global_features_to_combiner: bool,
            movie_width: int,
            movie_height: int):
    """Denoises a single frame from a single dataset"""
    
    # set to evaluation mode
    per_frame_feature_extractor.eval()
    temporal_combiner.eval()
    
    with torch.no_grad():
        
        # flatten across batch index and time into one 
        input_ntfxy = denoising_input_data['feature_enriched_occluded_input_ntfxy']
        n_batch, t_order, n_input_features, width, height = input_ntfxy.shape
        flat_input_features_mfxy = input_ntfxy.view(
            n_batch * t_order, n_input_features, width, height)

        # process each batch and frame
        frame_features_mlxy = per_frame_feature_extractor(flat_input_features_mfxy)

        # generate input for the combiner
        n_global_features = ws_denoising_list[0].features_1fxy.shape[-3]
        global_features_f_begin_index = 0
        global_features_f_end_index = n_global_features
        global_features_nfxy = crop_center(
            input_ntfxy[:, 0, global_features_f_begin_index:global_features_f_end_index, ...],
            target_width=frame_features_mlxy.shape[-2],
            target_height=frame_features_mlxy.shape[-1])

        t_stride = per_frame_feature_extractor.out_channels
        assert t_stride == 1 # let's be conservative for now -- talk to Luca
        frame_features_nqxy = frame_features_mlxy.view(
            (n_batch,
             t_order * t_stride,
             frame_features_mlxy.shape[-2],
             frame_features_mlxy.shape[-1]))

        if pass_global_features_to_combiner:
            combiner_input_ncxy = torch.cat(
                (frame_features_nqxy, global_features_nfxy), dim=-3)
        else:
            combiner_input_ncxy = frame_features_nqxy
        
        # combine
        combiner_output_nxy = temporal_combiner(combiner_input_ncxy)[:, 0, ...]

        return crop_center(
            combiner_output_nxy,
            target_width=movie_width,
            target_height=movie_height).mean(0)
    

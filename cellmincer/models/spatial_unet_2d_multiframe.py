import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union

from torchinfo import summary

from .components import \
    activation_from_str, \
    GUNet, \
    get_unet_input_size

from .denoising_model import DenoisingModel

from cellmincer.util.ws import OptopatchDenoisingWorkspace


class SpatialUnet2dMultiframe(DenoisingModel):
    def __init__(
            self,
            config: dict,
            device: torch.device,
            dtype: torch.dtype):
        
        super(SpatialUnet2dMultiframe, self).__init__(
            name=config['type'],
            t_order=config['t_order'],
            device=device,
            dtype=dtype)
        
        self.feature_mode = config['unet_feature_mode']
        assert self.feature_mode in {'repeat', 'once', 'none'}
        
        self.unet = GUNet(
            in_channels=self.t_order,
            out_channels=1,
            data_dim=2,
            feature_channels=config['n_global_features'],
            noise_channels=0,
            depth=config['unet_depth'],
            first_conv_channels=config['unet_first_conv_channels'],
            ch_growth_rate=2,
            ds_rate=2,
            final_trans=lambda x: x,
            pad=config['use_padding'],
            layer_norm=config['use_layer_norm'],
            attention=config['use_attention'],
            feature_mode=config['unet_feature_mode'],
            up_mode='upsample',
            pool_mode='max',
            norm_mode='batch',
            kernel_size=config['unet_kernel_size'],
            n_conv_layers=config['unet_n_conv_layers'],
            p_dropout=0.0,
            readout_hidden_layer_channels_list=[config['unet_first_conv_channels']],
            readout_kernel_size=config['unet_readout_kernel_size'],
            activation=activation_from_str(config['unet_activation']),
            device=device,
            dtype=dtype)
    
    def forward(
            self,
            x: torch.Tensor,
            features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        t_total = x.shape[1]
        t_tandem = t_total - self.t_order

        # calculate processed features
        unet_endpoint_ntxy = torch.cat([(
                self.unet(x[:, i_t:i_t+self.t_order, :, :], features) if self.feature_mode != 'none' else
                self.unet(x[:, i_t:i_t+self.t_order, :, :]))['readout']
            for i_t in range(t_tandem + 1)], dim=1)
        
        return unet_endpoint_ntxy
    
    def denoise_movie(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            t_begin: int = 0,
            t_end: int = None,
            x0: int = 0,
            y0: int = 0,
            x_window: int = None,
            y_window: int = None
        ) -> torch.Tensor:
        
        # defaults bounds to full movie if unspecified
        if t_end is None:
            t_end = ws_denoising.n_frames
        if x_window is None:
            x_window = ws_denoising.width - x0
        if y_window is None:
            y_window = ws_denoising.height - y0
        
        assert t_end - t_begin >= self.t_order
        assert 0 <= x0 <= x0 + x_window <= ws_denoising.width
        assert 0 <= y0 <= y0 + y_window <= ws_denoising.height
        
        n_frames = ws_denoising.n_frames
        t_mid = (self.t_order - 1) // 2
        mid_frame_begin = max(t_begin, t_mid)
        mid_frame_end = min(t_end, n_frames - t_mid)
        
        denoised_movie_txy_list = []
        
        if self.feature_mode != 'none':
            padded_global_features_1fxy = ws_denoising.get_feature_slice(
                x0=x0,
                y0=y0,
                x_window=x_window,
                y_window=y_window)
        
        with torch.no_grad():
            for i_t in range(mid_frame_begin - t_mid, mid_frame_end - t_mid):
                padded_sliced_movie_1txy = ws_denoising.get_movie_slice(
                    t_begin_index=i_t,
                    t_end_index=i_t + self.t_order,
                    x0=x0,
                    y0=y0,
                    x_window=x_window,
                    y_window=y_window)['diff']
                
                denoised_movie_txy_list.append(
                    (self.unet(padded_sliced_movie_1txy, padded_global_features_1fxy)
                        if self.feature_mode != 'none' else
                        self.unet(padded_sliced_movie_1txy))['readout'][0, ...].cpu())
        
        # fill in edge frames with the ends of the middle frame interval
        denoised_movie_txy_full_list = \
            [denoised_movie_txy_list[0] for i in range(mid_frame_begin - t_begin)] + \
            denoised_movie_txy_list + \
            [denoised_movie_txy_list[-1] for i in range(t_end - mid_frame_end)]
        
        denoised_movie_txy = torch.cat(denoised_movie_txy_full_list, dim=0)
        return denoised_movie_txy
    
    def summary(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            x_window: int,
            y_window: int):
        x_padding = self.get_window_padding(x_window)
        y_padding = self.get_window_padding(y_window)
        
        input_data = {}
        
        input_data['x'] = ws_denoising.get_movie_slice(
            t_begin_index=0,
            t_end_index=self.t_order,
            x0=0,
            y0=0,
            x_window=x_window,
            y_window=y_window,
            x_padding=x_padding,
            y_padding=y_padding)['diff']
        
        if self.feature_mode != 'none':
            input_data['features'] = ws_denoising.get_feature_slice(
                x0=0,
                y0=0,
                x_window=x_window,
                y_window=y_window,
                x_padding=x_padding,
                y_padding=y_padding)

        return str(summary(self, input_data=input_data))
    
    def get_window_padding(
            self,
            output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
        output_min_size = np.array(output_min_size)
        input_size = get_unet_input_size(
            output_min_size=output_min_size,
            kernel_size=self.unet.kernel_size,
            n_conv_layers=self.unet.n_conv_layers,
            depth=self.unet.depth,
            ds_rate=self.unet.ds_rate)
        padding = ((input_size - output_min_size) // 2).astype('int')
        return padding

    @staticmethod
    def get_window_padding_from_config(
            config: dict,
            output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
        output_min_size = np.array(output_min_size)
        input_size = get_unet_input_size(
            output_min_size=output_min_size,
            kernel_size=config['unet_kernel_size'],
            n_conv_layers=config['unet_n_conv_layers'],
            depth=config['unet_depth'],
            ds_rate=2)
        padding = ((input_size - output_min_size) // 2).astype('int')
        return padding

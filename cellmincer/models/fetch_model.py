import numpy as np
import torch
from torch import nn

import logging
from typing import Tuple, Union

from .denoising_model import DenoisingModel

# import model subclasses here...
from .spatial_unet_2d_temporal_denoiser import SpatialUnet2dTemporalDenoiser
from .spatial_unet_2d_multiframe import SpatialUnet2dMultiframe
from .spatiotemporal_unet_3d import SpatiotemporalUnet3d


# ...and add models to this lookup dictionary
_MODEL_DICT = {
    'spatial-unet-2d-temporal-denoiser': SpatialUnet2dTemporalDenoiser,
    'spatial-unet-2d-multiframe': SpatialUnet2dMultiframe,
    'spatiotemporal-unet-3d': SpatiotemporalUnet3d
}


def init_model(
        model_config: dict,
        device: torch.device = torch.device('cuda'),
        dtype: torch.dtype = torch.float32) -> DenoisingModel:
    try:
        denoising_model = _MODEL_DICT[model_config['type']](model_config, device, dtype)
    except KeyError:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)
        
    return denoising_model


def get_window_padding_from_config(
        model_config: dict,
        output_min_size: Union[int, list, np.ndarray]) -> np.ndarray:
    try:
        return _MODEL_DICT[model_config['type']].get_window_padding_from_config(
            config=model_config,
            output_min_size=output_min_size)
    except KeyError:
        logging.warning(f'Unrecognized model type; options are {", ".join([name for name in _MODEL_DICT])}')
        exit(0)
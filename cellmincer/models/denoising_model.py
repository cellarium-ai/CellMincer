import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

import logging
from typing import Tuple, Union

from cellmincer.util.ws import OptopatchBaseWorkspace, OptopatchDenoisingWorkspace


class DenoisingModel(pl.LightningModule):
    '''
    The base class for a CellMincer denoising model, extended from the standard PyTorch Lightning module.

    It defines methods for denoising recordings of arbitrary lengths, as well as supplying its temporal window length and padding expectations for use in training and inference.
    '''
    
    def __init__(
            self,
            name: str,
            t_order: int):
        if t_order is None:
            logging.warning('t_order assignment deferred; ensure this behavior is intended')
            logging.info(f'name: {name}')
        else:
            assert t_order & 1 == 1
            logging.info(f'name: {name} | t_order: {t_order}')
        
        super(DenoisingModel, self).__init__()
        self.name = name
        self.t_order = t_order

    '''
    Denoises the 'diff' movie segment in ws_denoising, bounded by [t_begin, t_end) and windowed by (x0, y0, x_window, y_window), returning a CPU tensor containing the denoised movie.
    
    :param ws_denoising: The workspace containing the movie to denoise.
    :param t_begin: The index of the first frame in the denoising segment (inclusive), defaults to 0.
    :param t_end: The index of the last frame in the denoising segment (exclusive), defaults to the end of the movie.
    :param x0: The x-coordinate of the top-left corner of the denoising window, defaults to 0.
    :param y0: The y-coordinate of the top-left corner of the denoising window, defaults to 0.
    :param x_window: The width of the denoising window, defaults to the full width.
    :param y_window: The height of the denoising window, defaults to the full height.
    '''
    def denoise_movie(
            self,
            ws_denoising: OptopatchDenoisingWorkspace,
            t_begin: int = 0,
            t_end: int = None,
            x0: int = 0,
            y0: int = 0,
            x_window: int = None,
            y_window: int = None) -> torch.Tensor:
        raise NotImplementedError

    '''
    Returns a summary of network architecture as a string representation.
    
    :param ws_denoising: The workspace containing the movie used as a sample input.
    '''
    def summary(self, ws_denoising: OptopatchDenoisingWorkspace):
        raise NotImplementedError

    '''
    Returns the padding size for an image dimension with given minimum output size.
    
    :param output_min_size: The minimum width or height expected as an output dimension.
    '''
    @staticmethod
    def get_window_padding(
            output_min_size: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError

    '''
    Returns the temporal order of a model with given configuration.
    
    :param config: A dictionary containing the model parameters.
    '''
    @staticmethod
    def get_temporal_order_from_config(config: dict) -> int:
        raise NotImplementedError

    '''
    Returns the padding size for an image dimension with given minimum output size and model configuration.
    
    :param config: A dictionary containing the model parameters.
    :param output_min_size: The minimum width or height expected as an output dimension.
    '''
    @staticmethod
    def get_window_padding_from_config(
            config: dict,
            output_min_size: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        raise NotImplementedError

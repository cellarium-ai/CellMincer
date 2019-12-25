import numpy as np
import torch
from typing import Optional


def get_cosine_similarity_with_sequence_np(
    movie_txy: np.ndarray,
    seq_t: np.ndarray,
    t_indices: Optional[np.ndarray] = None):
    """Calculates the pixel-wise cosine similarity between the zero-mean movie and
    a given temporal sequence.    
    """
    if t_indices is not None:
        seq_t = seq_t[t_indices]
        movie_txy = movie_txy[t_indices, ...]

    return np.einsum("txy,t->xy", movie_txy, seq_t) / (
        np.linalg.norm(movie_txy, axis=0) * np.linalg.norm(seq_t))


def get_cosine_similarity_with_sequence_torch(
    movie_txy: torch.tensor,
    seq_t: torch.Tensor,
    t_indices: Optional[torch.LongTensor] = None):
    """Calculates the pixel-wise cosine similarity between the zero-mean movie and
    a given temporal sequence.    
    """
    if t_indices is not None:
        seq_t = seq_t[t_indices]
        movie_txy = movie_txy[t_indices, ...]

    return torch.einsum("txy,t->xy", movie_txy, seq_t) / (
        torch.norm(movie_txy, dim=0) * torch.norm(seq_t))


def pad_images_np(images_ncxy: np.ndarray,
                  target_width: int,
                  target_height: int,
                  pad_value_nc: Optional[np.ndarray] = None) -> np.ndarray:
    assert images_ncxy.ndim == 4
    source_width, source_height = images_ncxy.shape[2], images_ncxy.shape[3]
    assert target_width >= source_width
    assert target_height >= source_height
    assert (target_width - source_width) % 2 == 0
    assert (target_height - source_height) % 2 == 0
    
    margin_width = (target_width - source_width) // 2
    margin_height = (target_height - source_height) // 2
    if pad_value_nc is None:
        pad_value_nc = np.mean(images_ncxy, axis=(-1, -2))
    output = np.tile(pad_value_nc[..., None, None], (1, 1, target_width, target_height))
    output[:, :,
           margin_width:(source_width + margin_width),
           margin_height:(source_height + margin_height)] = images_ncxy
    return output


def pad_images_torch(images_ncxy: torch.Tensor,
                     target_width: int,
                     target_height: int,
                     pad_value_nc: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert images_ncxy.ndim == 4
    source_width, source_height = images_ncxy.shape[2], images_ncxy.shape[3]
    assert target_width >= source_width
    assert target_height >= source_height
    assert (target_width - source_width) % 2 == 0
    assert (target_height - source_height) % 2 == 0
    
    margin_width = (target_width - source_width) // 2
    margin_height = (target_height - source_height) // 2
    if pad_value_nc is None:
        pad_value_nc = torch.mean(images_ncxy, dim=(-1, -2))
    
    output = pad_value_nc[..., None, None].repeat(1, 1, target_width, target_height)
    output[:, :,
           margin_width:(source_width + margin_width),
           margin_height:(source_height + margin_height)] = images_ncxy
    return output


def crop_center(input_ncxy: torch.Tensor,
                target_width: int,
                target_height: int) -> torch.Tensor:
    input_width = input_ncxy.shape[-2]
    input_height = input_ncxy.shape[-1]
    margin_width = (input_width - target_width) // 2
    margin_height = (input_height - target_height) // 2
    return input_ncxy[...,
           margin_width:(target_width + margin_width),
           margin_height:(target_height + margin_height)]


def get_nn_spatio_temporal_mean(movie_ntxy: torch.Tensor, i_t: int) -> torch.Tensor:
    ONE_OVER_TWENTY_SIX = 0.03846153846
    return ONE_OVER_TWENTY_SIX * (
        movie_ntxy[..., i_t, 0:-2, 0:-2]
            + movie_ntxy[..., i_t, 0:-2, 1:-1]
            + movie_ntxy[..., i_t, 0:-2, 2:]
            + movie_ntxy[..., i_t, 1:-1, 0:-2]
            + movie_ntxy[..., i_t, 1:-1, 2:]
            + movie_ntxy[..., i_t, 2:, 0:-2]
            + movie_ntxy[..., i_t, 2:, 1:-1]
            + movie_ntxy[..., i_t, 2:, 2:]
            + movie_ntxy[..., i_t - 1, 0:-2, 0:-2]
            + movie_ntxy[..., i_t - 1, 0:-2, 1:-1]
            + movie_ntxy[..., i_t - 1, 0:-2, 2:]
            + movie_ntxy[..., i_t - 1, 1:-1, 0:-2]
            + movie_ntxy[..., i_t - 1, 1:-1, 1:-1]
            + movie_ntxy[..., i_t - 1, 1:-1, 2:]
            + movie_ntxy[..., i_t - 1, 2:, 0:-2]
            + movie_ntxy[..., i_t - 1, 2:, 1:-1]
            + movie_ntxy[..., i_t - 1, 2:, 2:]
            + movie_ntxy[..., i_t + 1, 0:-2, 0:-2]
            + movie_ntxy[..., i_t + 1, 0:-2, 1:-1]
            + movie_ntxy[..., i_t + 1, 0:-2, 2:]
            + movie_ntxy[..., i_t + 1, 1:-1, 0:-2]
            + movie_ntxy[..., i_t + 1, 1:-1, 1:-1]
            + movie_ntxy[..., i_t + 1, 1:-1, 2:]
            + movie_ntxy[..., i_t + 1, 2:, 0:-2]
            + movie_ntxy[..., i_t + 1, 2:, 1:-1]
            + movie_ntxy[..., i_t + 1, 2:, 2:])


import numpy as np
from skimage.filters import threshold_otsu
from boltons.cacheutils import cachedproperty
import torch
import logging
from typing import List, Tuple, Optional

from .opto_utils import get_cosine_similarity_with_sequence_np, pad_images_np

logger = logging.getLogger()


class OptopatchBaseWorkspace:
    """Workspace for caching useful quantities"""
    
    EPS = 1e-6
    DEFAULT_NEIGHBOR_DX_DY_LIST = [
        (1, 1), (1, 0), (1, -1),
        (0, 1), (0, -1),
        (-1, 1), (-1, 0), (-1, -1)]

    def __init__(self,
                 opto_mov_path: str,
                 logger: logging.Logger,
                 dtype = np.float32,
                 neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        
        self.opto_mov_path = opto_mov_path
        self.logger = logger
        self.dtype = dtype
        self.neighbor_dx_dy_list = neighbor_dx_dy_list
        
        # load the movie
        self.info(f"Loading movie from {opto_mov_path} ...")
        self.movie_txy = np.load(opto_mov_path).transpose(-3, -1, -2)

    def info(self, msg: str):
        self.logger.warning(msg)
        
    @cachedproperty
    def n_frames(self):
        return self.movie_txy.shape[-3]
    
    @cachedproperty
    def width(self):
        return self.movie_txy.shape[-2]

    @cachedproperty
    def height(self):
        return self.movie_txy.shape[-1]

    @cachedproperty
    def n_pixels(self):
        return self.width * self.height

    @cachedproperty
    def movie_t_mean_xy(self):
        """Temporal mean"""
        self.info("Calculating temporal mean ...")
        return np.mean(self.movie_txy, 0).astype(self.dtype)
    
    @cachedproperty
    def movie_t_std_xy(self):
        """Temporal std"""
        self.info("Calculating temporal std ...")
        return np.std(self.movie_txy, 0).astype(self.dtype)
    
    @property
    def movie_zero_mean_txy(self):
        """Temporal zero-mean movie"""
        self.info("Calculating zero-mean movie ...")
        return (self.movie_txy - self.movie_t_mean_xy[None, ...]).astype(self.dtype)
    
    @cachedproperty
    def movie_t_corr_xy_list(self) -> List[np.ndarray]:
        """Peason correlation with nearest neighobrs"""
        self.info("Calculating temporal correlation with neighbors ...")
        movie_t_corr_xy_list = []
        movie_zero_mean_txy = self.movie_zero_mean_txy
        for dx, dy in self.neighbor_dx_dy_list:
            # calculate Pearson correlation witrh a neighbor
            movie_zero_mean_dxdy_txy = np.roll(
                movie_zero_mean_txy,
                shift=(dx, dy),
                axis=(-2, -1))
            movie_t_std_dxdy_xy = np.roll(
                self.movie_t_std_xy,
                shift=(dx, dy),
                axis=(-2, -1))
            movie_t_corr_dxdy_xy = np.einsum("txy,txy->xy",
                movie_zero_mean_txy, movie_zero_mean_dxdy_txy) / (
                self.EPS + self.n_frames * self.movie_t_std_xy * movie_t_std_dxdy_xy)
            movie_t_corr_xy_list.append(movie_t_corr_dxdy_xy.astype(self.dtype))
        return movie_t_corr_xy_list
    
    @cachedproperty
    def movie_t_corr_xy(self) -> np.ndarray:
        return np.maximum.reduce(self.movie_t_corr_xy_list).astype(self.dtype)
    
    @cachedproperty
    def corr_otsu_threshold(self) -> float:
        return threshold_otsu(self.movie_t_corr_xy)

    @cachedproperty
    def corr_otsu_fg_pixel_mask_xy(self) -> np.ndarray:
        return self.movie_t_corr_xy >= self.corr_otsu_threshold
    
    @cachedproperty
    def corr_otsu_fg_weight(self) -> float:
        return self.corr_otsu_fg_pixel_mask_xy.sum().item() / self.n_pixels

    @cachedproperty
    def corr_otsu_fg_mean_t(self) -> np.ndarray:
        return np.mean(
            self.movie_txy.reshape(self.n_frames, -1)[:, self.corr_otsu_fg_pixel_mask_xy.reshape(-1)],
            axis=-1).astype(self.dtype)
    
    @cachedproperty
    def movie_cosine_fg_sim_xy(self) -> np.ndarray:
        return get_cosine_similarity_with_sequence_np(
            self.movie_txy, self.corr_otsu_fg_mean_t).astype(self.dtype)
    
    @cachedproperty
    def cosine_fg_sim_otsu_threshold(self) -> float:
        return threshold_otsu(self.movie_cosine_fg_sim_xy)

    @cachedproperty
    def cosine_fg_sim_otsu_fg_pixel_mask_xy(self) -> np.ndarray:
        return self.movie_cosine_fg_sim_xy >= self.cosine_fg_sim_otsu_threshold
    
    @cachedproperty
    def cosine_fg_sim_otsu_fg_weight(self) -> float:
        return self.cosine_fg_sim_otsu_fg_pixel_mask_xy.sum().item() / self.n_pixels


class OptopatchDenoisingWorkspace:
    """A workspace containing arrays prepared for denoising (e.g. normalized, padded)"""
    def __init__(self,
                 ws_base: OptopatchBaseWorkspace,
                 target_width: int,
                 target_height: int,
                 dtype=np.float32):
        self.ws_base = ws_base
        self.target_width = target_width
        self.target_height = target_height
        self.dtype = dtype
        
        # estimate fg scale for normalization
        fg_scale = ws_base.movie_t_std_xy[ws_base.cosine_fg_sim_otsu_fg_pixel_mask_xy].mean()
        self.fg_scale = fg_scale
        
        # pad the scaled movie
        self.padded_movie_1txy = pad_images_np(
            images_ncxy=ws_base.movie_txy[None, ...] / fg_scale,
            target_width=target_width,
            target_height=target_height).astype(dtype)

        # generate global features
        features = []
        feature_names = []
        
        # std
        padded_movie_std_11xy_scaled = pad_images_np(
            images_ncxy=ws_base.movie_t_std_xy[None, None, :, :] / fg_scale,
            target_width=target_width,
            target_height=target_height)
        features.append(padded_movie_std_11xy_scaled)
        feature_names.append('std')

        # mean
        padded_movie_mean_11xy_scaled = pad_images_np(
            images_ncxy=ws_base.movie_t_mean_xy[None, None, :, :] / fg_scale,
            target_width=target_width,
            target_height=target_height)
        features.append(padded_movie_mean_11xy_scaled)
        feature_names.append('mean')

        # fg cosine sim
        padded_movie_cosine_fg_sim_11xy_scaled = pad_images_np(
            images_ncxy=ws_base.movie_cosine_fg_sim_xy[None, None, :, :],
            target_width=target_width,
            target_height=target_height)
        features.append(padded_movie_cosine_fg_sim_11xy_scaled)
        feature_names.append('cosine_fg_sim')
        
        # kNN temporal correlations
        for (dx, dy), t_corr_xy in zip(
                ws_base.neighbor_dx_dy_list, ws_base.movie_t_corr_xy_list):
            padded_t_corr_11xy_scaled = pad_images_np(
                images_ncxy=t_corr_xy[None, None, :, :],
                target_width=target_width,
                target_height=target_height)
            features.append(padded_t_corr_11xy_scaled)
            feature_names.append(f't_corr_({dx}, {dy})')
        
        # concatenate gobal features
        self.features_1fxy = np.concatenate(features, -3).astype(dtype)
        self.feature_names = feature_names
        
    def get_slice(self, begin_t_index: int, end_t_index: int) -> np.ndarray:
        return self.padded_movie_1txy[:, begin_t_index:end_t_index, ...]

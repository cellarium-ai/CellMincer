import numpy as np
from skimage.filters import threshold_otsu
from boltons.cacheutils import cachedproperty
import torch
import tifffile
import logging
from typing import Dict, List, Optional, Tuple, Union

from .utils import get_cosine_similarity_with_sequence_np
from .features import OptopatchGlobalFeatureContainer

from . import const


class OptopatchBaseWorkspace:
    """Workspace for caching useful quantities"""

    DEFAULT_NEIGHBOR_DX_DY_LIST = [
        (1, 1), (1, 0), (1, -1),
        (0, 1), (0, -1),
        (-1, 1), (-1, 0), (-1, -1)]

    def __init__(self,
                 movie_txy: np.ndarray,
                 dtype = np.float32,
                 neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        self.movie_txy = movie_txy.astype(dtype)
        self.dtype = dtype
        self.neighbor_dx_dy_list = neighbor_dx_dy_list

    @staticmethod
    def from_bin_uint16(
            movie_bin_path: str,
            n_frames: int,
            width: int,
            height: int,
            order: str = 'txy',
            dtype = np.float32,
            neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        # load the movie
        logging.debug(f"Loading movie from {movie_bin_path} ...")
        shape_dict = {'x': width, 'y': height, 't': n_frames}
        shape = tuple(map(shape_dict.get, order))
        movie_nnn = np.fromfile(movie_bin_path, dtype=np.uint16).reshape(shape, order='C')
        movie_txy = movie_nnn.transpose(tuple(map(order.find, 'txy')))
        return OptopatchBaseWorkspace(
            movie_txy=movie_txy,
            dtype=dtype,
            neighbor_dx_dy_list=neighbor_dx_dy_list) 

    @staticmethod
    def from_npy(
            movie_npy_path: str,
            order: str = 'txy',
            dtype = np.float32,
            neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        # load the movie
        logging.debug(f"Loading movie from {movie_npy_path} ...")
        movie_nnn = np.load(movie_npy_path).astype(dtype)
        movie_txy = movie_nnn.transpose(tuple(map(order.find, 'txy')))
        return OptopatchBaseWorkspace(
            movie_txy=movie_txy,
            dtype=dtype,
            neighbor_dx_dy_list=neighbor_dx_dy_list)
    
    @staticmethod
    def from_npz(
            movie_npz_path: str,
            order: str = 'txy',
            key = 'arr_0',
            dtype = np.float32,
            neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        # load the movie
        logging.debug(f"Loading movie from {movie_npz_path} ...")
        
        npz_file = np.load(movie_npz_path)
        movie_nnn = npz_file[key].astype(dtype)
        npz_file.close()
        movie_txy = movie_nnn.transpose(tuple(map(order.find, 'txy')))
        return OptopatchBaseWorkspace(
            movie_txy=movie_txy,
            dtype=dtype,
            neighbor_dx_dy_list=neighbor_dx_dy_list)
    
    @staticmethod
    def from_tiff(
            movie_tiff_path: str,
            order: str = 'txy',
            dtype = np.float32,
            neighbor_dx_dy_list: List[Tuple[int, int]] = DEFAULT_NEIGHBOR_DX_DY_LIST):
        # load the movie
        logging.debug(f"Loading movie from {movie_tiff_path} ...")
        movie_nnn = tifffile.imread(movie_tiff_path).astype(dtype)
        movie_txy = movie_nnn.transpose(tuple(map(order.find, 'txy')))
        return OptopatchBaseWorkspace(
            movie_txy=movie_txy,
            dtype=dtype,
            neighbor_dx_dy_list=neighbor_dx_dy_list)

    def get_t_truncated_movie(self, t_mask: np.ndarray) -> 'OptopatchBaseWorkspace':
        return OptopatchBaseWorkspace(
            movie_txy=self.movie_txy[t_mask, :, :],
            dtype=self.dtype,
            neighbor_dx_dy_list=self.neighbor_dx_dy_list)
    
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
        logging.debug("Calculating temporal mean ...")
        return np.mean(self.movie_txy, 0).astype(self.dtype)
    
    @cachedproperty
    def movie_t_std_xy(self):
        """Temporal std"""
        logging.debug("Calculating temporal std ...")
        return np.std(self.movie_txy, 0).astype(self.dtype)
    
    @property
    def movie_zero_mean_txy(self):
        """Temporal zero-mean movie"""
        logging.debug("Calculating zero-mean movie ...")
        return (self.movie_txy - self.movie_t_mean_xy[None, ...]).astype(self.dtype)
    
    @cachedproperty
    def movie_t_corr_xy_list(self) -> List[np.ndarray]:
        """Peason correlation with nearest neighobrs"""
        logging.debug("Calculating temporal correlation with neighbors ...")
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
                const.EPS + self.n_frames * self.movie_t_std_xy * movie_t_std_dxdy_xy)
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


class OptopatchGlobalFeaturesTorchCache:
    def __init__(
            self,
            features: OptopatchGlobalFeatureContainer,
            x_padding: int,
            y_padding: int,
            padding_mode: Optional[str] = 'reflect',
            device: torch.device = const.DEFAULT_DEVICE,
            dtype: torch.dtype = const.DEFAULT_DTYPE):
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.device = device
        self.dtype = dtype
        
        self.features_1fxy = torch.tensor(
            np.concatenate([
                np.pad(
                    array=feature_array_xy,
                    pad_width=((x_padding, x_padding), (y_padding, y_padding)),
                    mode=padding_mode)[None, None, ...]
                for feature_array_xy in features.feature_array_list],
                axis=-3),
            device=device,
            dtype=dtype)
        
        self.norm_scale = features.norm_scale
        self.feature_name_list = features.feature_name_list
        self.feature_depth_list = features.feature_depth_list
        self.feature_name_to_idx_map = {
            self.feature_name_list[idx]: idx
            for idx in range(len(self.feature_name_list))}

    def get_feature_index(self, feature_name: str):
        return self.feature_name_to_idx_map[feature_name]


class OptopatchDenoisingWorkspace:
    """A workspace containing arrays prepared for denoising (e.g. normalized, padded)"""
    def __init__(self,
                 movie_diff: np.ndarray,
                 movie_bg: Union[np.ndarray, None],
                 noise_params: dict,
                 features: OptopatchGlobalFeatureContainer,
                 x_padding: int,
                 y_padding: int,
                 padding_mode: Optional[str] = 'reflect',
                 occlude_padding: Optional[bool] = False,
                 device: torch.device = const.DEFAULT_DEVICE,
                 dtype: torch.dtype = const.DEFAULT_DTYPE):
        self.noise_params = noise_params
        
        self.n_frames, self.width, self.height = movie_diff.shape[-3:]
        
        assert x_padding > 0 and y_padding > 0
        
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.padded_width = self.width + 2 * x_padding
        self.padded_height = self.height + 2 * y_padding
        
        self.device = device
        self.dtype = dtype
        
        assert padding_mode in ('reflect', 'constant')

        # pad and cache the features
        self.cached_features = OptopatchGlobalFeaturesTorchCache(
            features=features,
            x_padding=x_padding,
            y_padding=y_padding,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)
        
        trend_mean_feature_index = self.cached_features.get_feature_index('trend_mean_0')
        detrended_std_feature_index = self.cached_features.get_feature_index('detrended_std_0')
        
        # pad the scaled movie with occluded pixels sampled with feature map
        if occlude_padding:
            occluded_padding_map_txy = torch.distributions.Normal(
                loc=self.cached_features.features_1fxy[0, trend_mean_feature_index, :, :],
                scale=self.cached_features.features_1fxy[0, detrended_std_feature_index, :, :] + const.EPS).sample((self.n_frames,)).cpu().numpy()

            occluded_padding_map_txy[:, x_padding:-x_padding, y_padding:-y_padding] = movie_diff / features.norm_scale
        
            self.padded_scaled_diff_movie_1txy = occluded_padding_map_txy[None, ...]

        else:
            self.padded_scaled_diff_movie_1txy = np.pad(
                array=movie_diff / features.norm_scale,
                pad_width=((0, 0), (x_padding, x_padding), (y_padding, y_padding)),
                mode=padding_mode)[None, ...]
        
        self.padded_scaled_bg_movie_1txy = np.pad(
            array=movie_bg / features.norm_scale,
            pad_width=((0, 0), (x_padding, x_padding), (y_padding, y_padding)),
            mode=padding_mode)[None, ...] if movie_bg is not None else None
        
    def get_movie_slice(
            self,
            t_begin_index: int,
            t_end_index: int,
            x0: int,
            y0: int,
            x_window: int,
            y_window: int,
            x_padding: Optional[int] = None,
            y_padding: Optional[int] = None) -> Dict[str, torch.Tensor]:
        assert self.width - x_window >= 0
        assert self.height - y_window >= 0
        assert 0 <= x0 <= self.width - x_window
        assert 0 <= y0 <= self.height - y_window
        assert 0 <= t_begin_index <= self.n_frames
        assert 0 <= t_end_index <= self.n_frames
        assert t_end_index > t_begin_index
        
        if x_padding is None:
            x_padding = self.x_padding
        if y_padding is None:
            y_padding = self.y_padding
        
        diff_movie_slice_1txy = torch.tensor(
            self.padded_scaled_diff_movie_1txy[:, t_begin_index:t_end_index, ...][
                ...,
                x0:(x0 + x_window + 2 * x_padding),
                y0:(y0 + y_window + 2 * y_padding)],
            device=self.device,
            dtype=self.dtype)
        
        bg_movie_slice_1txy = torch.tensor(
            self.padded_scaled_bg_movie_1txy[:, t_begin_index:t_end_index, ...][
                ...,
                x0:(x0 + x_window + 2 * x_padding),
                y0:(y0 + y_window + 2 * y_padding)],
            device=self.device,
            dtype=self.dtype) if self.padded_scaled_bg_movie_1txy else None

        return {
            'bg': bg_movie_slice_1txy,
            'diff': diff_movie_slice_1txy
        }
    
    def get_feature_slice(
            self,
            x0: int,
            y0: int,
            x_window: int,
            y_window: int,
            x_padding: Optional[int] = None,
            y_padding: Optional[int] = None) -> torch.Tensor:
        assert self.width - x_window >= 0
        assert self.height - y_window >= 0
        assert 0 <= x0 <= self.width - x_window
        assert 0 <= y0 <= self.height - y_window
        
        if x_padding is None:
            x_padding = self.x_padding
        if y_padding is None:
            y_padding = self.y_padding
        
        feature_slice_1fxy = self.cached_features.features_1fxy[
            :, :,
            x0:(x0 + x_window + 2 * x_padding),
            y0:(y0 + y_window + 2 * y_padding)]
        
        return feature_slice_1fxy
    
    def get_modeled_variance(
            self,
            scaled_bg_movie_txy: torch.Tensor,
            scaled_diff_movie_txy: torch.Tensor) -> torch.Tensor:
        if self.noise_params is None:
            raise Exception('Cannot model variance with undefined noise parameters')
        
        s = self.cached_features.norm_scale
        var_ntxy = torch.clamp(
            (self.noise_params['alpha_median'] * s * (scaled_bg_movie_txy + scaled_diff_movie_txy)
             + self.noise_params['beta_median']),
            min=self.noise_params['global_min_variance']) / (s ** 2)
        return var_ntxy

    @property
    def n_global_features(self):
        return self.cached_features.features_1fxy.shape[-3]

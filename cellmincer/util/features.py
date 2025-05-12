import numpy as np
import torch
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from skimage.filters import threshold_otsu
from scipy.signal import find_peaks


from .utils import crop_center

from . import const

@dataclass
class PaddedMovieTorch:
    t_padding: int
    x_padding: int
    y_padding: int
    original_n_frames: int
    original_width: int
    original_height: int
    padded_movie_txy: torch.Tensor
        
    def mean_xy(self):
        return self.padded_movie_txy.mean(0)
    
    def std_xy(self):
        return self.padded_movie_txy.std(0)


@dataclass
class OptopatchGlobalFeatureContainer:
    norm_scale: float = 1.0
    feature_name_list: List[str] = field(default_factory=list)
    feature_depth_list: List[int] = field(default_factory=list)
    feature_array_list: List[np.ndarray] = field(default_factory=list)


def unpad_frame(frame_xy: torch.Tensor, padded_movie: PaddedMovieTorch) -> torch.Tensor:
    return frame_xy[
        padded_movie.x_padding:(padded_movie.x_padding + padded_movie.original_width),
        padded_movie.y_padding:(padded_movie.y_padding + padded_movie.original_height)]

def is_power_of_two(x: int) -> bool:
    return (x & (x - 1) == 0) and x != 0


def smallest_power_of_two(x: int) -> int:
    return 2 ** (x-1).bit_length()


def generate_padded_movie(
        orig_movie_txy_np: np.ndarray,
        min_t_padding: int,
        min_x_padding: int,
        min_y_padding: int,
        padding_mode: str,
        power_of_two: bool,
        dtype: torch.dtype = const.DEFAULT_DTYPE) -> PaddedMovieTorch:
    
    original_n_frames = orig_movie_txy_np.shape[0]
    original_width = orig_movie_txy_np.shape[1]
    original_height = orig_movie_txy_np.shape[2]
    
    assert original_width % 2 == 0
    assert original_height % 2 == 0
    
    if power_of_two:
        assert is_power_of_two(original_width)
        assert is_power_of_two(original_height)
        x_padding = smallest_power_of_two(min_x_padding)
        y_padding = smallest_power_of_two(min_y_padding)
        t_padding = min_t_padding
    else:
        x_padding = min_x_padding
        y_padding = min_y_padding
        t_padding = min_t_padding

    padded_movie_txy_np = np.pad(
        orig_movie_txy_np,
        pad_width=((t_padding, t_padding), (x_padding, x_padding), (y_padding, y_padding)),
        mode=padding_mode)

    padded_movie_txy_torch = torch.tensor(
        padded_movie_txy_np,
        dtype=dtype)

    return PaddedMovieTorch(
        t_padding=t_padding,
        x_padding=x_padding,
        y_padding=y_padding,
        original_n_frames=original_n_frames,
        original_width=original_width,
        original_height=original_height,
        padded_movie_txy=padded_movie_txy_torch)


def get_trend_movie(
        padded_movie: PaddedMovieTorch,
        order: int,
        trend_func: str = 'mean') -> torch.Tensor:

    assert padded_movie.t_padding >= order
        
    trend_movie_txy = torch.zeros(
        (padded_movie.original_n_frames + 2 * padded_movie.t_padding - 2 * order,
         padded_movie.original_width + 2 * padded_movie.x_padding,
         padded_movie.original_height + 2 * padded_movie.y_padding),
        dtype=padded_movie.padded_movie_txy.dtype)

    # calculate temporal moving average
    if trend_func == 'mean':
        for i_t in range(trend_movie_txy.shape[0]):
            trend_movie_txy[i_t, ...] = torch.mean(
                padded_movie.padded_movie_txy[i_t:(i_t + 2 * order + 1), ...],
                dim=0)
    elif trend_func == 'median':
        for i_t in range(trend_movie_txy.shape[0]):
            trend_movie_txy[i_t, ...] = torch.median(
                padded_movie.padded_movie_txy[i_t:(i_t + 2 * order + 1), ...],
                dim=0)[0]

    return PaddedMovieTorch(
        t_padding=padded_movie.t_padding - order,
        x_padding=padded_movie.x_padding,
        y_padding=padded_movie.y_padding,
        original_n_frames=padded_movie.original_n_frames,
        original_width=padded_movie.original_width,
        original_height=padded_movie.original_height,
        padded_movie_txy=trend_movie_txy)


def calculate_cross(
        padded_movie: PaddedMovieTorch,
        trend_movie: Optional[PaddedMovieTorch],
        dt: int,
        dx: int,
        dy: int,
        normalize: bool) -> torch.Tensor:
    
    assert abs(dt) <= padded_movie.t_padding
    assert abs(dx) <= padded_movie.x_padding
    assert abs(dy) <= padded_movie.y_padding
    
    displaced_movie_txy = padded_movie.padded_movie_txy[
        (padded_movie.t_padding + dt):(padded_movie.t_padding + dt + padded_movie.original_n_frames),
        (padded_movie.x_padding + dx):(padded_movie.x_padding + dx + padded_movie.original_width),
        (padded_movie.y_padding + dy):(padded_movie.y_padding + dy + padded_movie.original_height)]
    
    original_movie_txy = padded_movie.padded_movie_txy[
        (padded_movie.t_padding):(padded_movie.t_padding + padded_movie.original_n_frames),
        (padded_movie.x_padding):(padded_movie.x_padding + padded_movie.original_width),
        (padded_movie.y_padding):(padded_movie.y_padding + padded_movie.original_height)]

    norm_xy = 1.

    # subtract trend
    if trend_movie is not None:
        displaced_trend_movie_txy = trend_movie.padded_movie_txy[
            (trend_movie.t_padding + dt):(trend_movie.t_padding + dt + trend_movie.original_n_frames),
            (trend_movie.x_padding + dx):(trend_movie.x_padding + dx + trend_movie.original_width),
            (trend_movie.y_padding + dy):(trend_movie.y_padding + dy + trend_movie.original_height)]
        
        original_trend_movie_txy = trend_movie.padded_movie_txy[
            (trend_movie.t_padding):(trend_movie.t_padding + trend_movie.original_n_frames),
            (trend_movie.x_padding):(trend_movie.x_padding + trend_movie.original_width),
            (trend_movie.y_padding):(trend_movie.y_padding + trend_movie.original_height)]
        
        cross_inner_xy = torch.mean(
            (displaced_movie_txy - displaced_movie_txy.mean(0)
             - displaced_trend_movie_txy + displaced_trend_movie_txy.mean(0)) *
            (original_movie_txy - original_movie_txy.mean(0)
             - original_trend_movie_txy + original_trend_movie_txy.mean(0)),
            dim=0)
        
        if normalize:
            norm_xy = (
                torch.std(displaced_movie_txy - displaced_trend_movie_txy, dim=0) * 
                torch.std(original_movie_txy - original_trend_movie_txy, dim=0))
        
    else:
        
        cross_inner_xy = torch.mean(
            (displaced_movie_txy - displaced_movie_txy.mean(0)) *
            (original_movie_txy - original_movie_txy.mean(0)),
            dim=0)
    
        if normalize:
            norm_xy = (
                torch.std(displaced_movie_txy, dim=0) * 
                torch.std(original_movie_txy, dim=0))

    
    return cross_inner_xy / norm_xy


def get_spatially_downsampled(
        padded_movie: PaddedMovieTorch,
        mode: str) -> PaddedMovieTorch:
    assert mode in {'max_pool', 'avg_pool'}
    
    assert padded_movie.x_padding % 2 == 0
    assert padded_movie.y_padding % 2 == 0
    assert padded_movie.original_width % 2 == 0
    assert padded_movie.original_height % 2 == 0
    
    if mode == 'avg_pool':
        downsampled_movie_txy = torch.nn.functional.avg_pool2d(
            padded_movie.padded_movie_txy.unsqueeze(0),
            kernel_size=2,
            stride=2).squeeze(0)
    elif mode == 'max_pool':
        downsampled_movie_txy = torch.nn.functional.max_pool2d(
            padded_movie.padded_movie_txy.unsqueeze(0),
            kernel_size=2,
            stride=2).squeeze(0)
    else:
        raise RuntimeError("Should not reach here!")
    
    return PaddedMovieTorch(
        t_padding=padded_movie.t_padding,
        x_padding=padded_movie.x_padding // 2,
        y_padding=padded_movie.y_padding // 2,
        original_n_frames=padded_movie.original_n_frames,
        original_width=padded_movie.original_width // 2,
        original_height=padded_movie.original_height // 2,
        padded_movie_txy=downsampled_movie_txy)


def upsample_to_numpy(frame_xy: torch.Tensor, depth: int):
    if depth == 0:
        return frame_xy.cpu().numpy()
    else:
        return torch.nn.functional.interpolate(
            frame_xy.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            align_corners=False,
            scale_factor=2).squeeze(0).squeeze(0).cpu().numpy()

    
class OptopatchGlobalFeatureExtractor:
    def __init__(
            self,
            movie_txy: np.ndarray,
            active_mask: Optional[np.ndarray],
            max_depth: int = 3,
            detrending_order: int = 10,
            trend_func: str = 'mean',
            downsampling_mode: str = 'avg_pool',
            padding_mode: str = 'reflect',
            dtype: torch.dtype = const.DEFAULT_DTYPE):
        
        self.movie_txy = movie_txy
        self.max_depth = max_depth
        self.detrending_order = detrending_order
        self.trend_func = trend_func
        self.downsampling_mode = downsampling_mode
        self.padding_mode = padding_mode
        self.dtype = dtype

        # containers
        self.active_mask_t = active_mask if active_mask is not None else self._infer_active_t_range(movie_txy)
        self.features = OptopatchGlobalFeatureContainer()

        # populate features
        self._populate_features()

    @staticmethod
    def _infer_active_t_range(
            movie_txy: np.ndarray,
            strategy: str = 'prominence',
            n_frame_smoothing: int = 7,
            prominence_bins: int = 100,
            prominence_threshold: float = 0.125):
        '''
        Identifies active frames of movie using an adaptation of foreground separation.
        
        The "otsu" strategy is recommended when the average intensity during activity is generally uniform throughout the movie.
        The "prominence" strategy is recommended when the movie exhibits varying levels of average activity, as when the stimulation intensity is varied.
        
        :param movie_txy: The movie over which activity is inferred.
        :param strategy: The strategy with which the activity threshold is computed.
        :param n_frame_smoothing: The width of the moving average window applied to the average intensity, for prominence calculation.
        :param prominence_bins: The number of bins over which average intensity is binned. Recommended to set large enough to resolve the separation between the background frames cluster and the weakest foreground frames cluster.
        :param prominence_threshold: Prominence threshold for histogram peak-finding, as a proportion of total histogram frequency range.
        '''
        mask_t = np.mean(movie_txy, axis=(-1, -2))
        
        assert strategy in ('otsu', 'prominence')

        if strategy == 'otsu':
            threshold = threshold_otsu(mask_t)
        elif strategy == 'prominence':
            before_pad = n_frame_smoothing // 2
            after_pad = n_frame_smoothing - before_pad - 1

            smooth_mask_t = np.convolve(np.pad(mask_t, (before_pad, after_pad), mode='edge'), np.ones(n_frame_smoothing) / n_frame_smoothing, mode='valid')
            m_histo, m_bins = np.histogram(smooth_mask_t, bins=prominence_bins)

            prominence = np.ptp(m_histo) * prominence_threshold

            peaks, _ = find_peaks(m_histo, prominence=prominence)
            threshold = (m_bins[peaks[0]] + m_bins[peaks[1]]) / 2
        logging.info(f'threshold: {threshold}')

        active_mask_t = OptopatchGlobalFeatureExtractor._get_continuous_1d_mask(mask_t > threshold)
        return active_mask_t

    @staticmethod
    def _get_continuous_1d_mask(mask_t: np.ndarray, n_frame_smoothing: int = 11) -> np.ndarray:
        smooth_mask_t = np.convolve(mask_t, np.ones(n_frame_smoothing) / n_frame_smoothing, mode='same')
        return smooth_mask_t > 0.5

    def _populate_features(self):
        # pad the original movie to power of two
        input_movie_txy = self.movie_txy[self.active_mask_t, ...]
        input_movie_std_scale = np.std(input_movie_txy)

        original_width = input_movie_txy.shape[-2]
        original_height = input_movie_txy.shape[-1]
        
        assert original_width % 2 == 0
        assert original_height % 2 == 0

        x_padding = (smallest_power_of_two(original_width) - original_width) // 2
        y_padding = (smallest_power_of_two(original_height) - original_height) // 2
        pow_two_padded_input_movie_txy = np.pad(
            input_movie_txy,
            pad_width=((0, 0), (x_padding, x_padding), (y_padding, y_padding)),
            mode=self.padding_mode)

        # pad additional for easy calculation of spatio-temporal cross-correlations
        padded_movie = generate_padded_movie(
            orig_movie_txy_np=pow_two_padded_input_movie_txy,
            min_t_padding=2 * self.detrending_order,
            min_x_padding=2 ** self.max_depth,
            min_y_padding=2 ** self.max_depth,
            padding_mode=self.padding_mode,
            power_of_two=True,
            dtype=self.dtype)

        # normalization scale
        self.features.norm_scale = input_movie_std_scale
        
        # neighbors to consider in calculating cross-correlations
        corr_displacement_list = []
        for dt in [0, 1]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (dt, dx, dy) != (0, 0, 0):
                        corr_displacement_list.append((dt, dx, dy))

        prev_padded_movie = padded_movie
        prev_trend_movie = get_trend_movie(
            padded_movie=prev_padded_movie,
            order=self.detrending_order,
            trend_func=self.trend_func)

        for depth in range(self.max_depth + 1):
            if depth > 0:
                logging.debug(f'Downsampling to depth {depth}...')
                current_padded_movie = get_spatially_downsampled(
                    padded_movie=prev_padded_movie,
                    mode=self.downsampling_mode)
                current_trend_movie = get_spatially_downsampled(
                    padded_movie=prev_trend_movie,
                    mode=self.downsampling_mode)
            else:
                current_padded_movie = prev_padded_movie
                current_trend_movie = prev_trend_movie

            # calculate detrended std
            current_detrended_var_xy = calculate_cross(
                padded_movie=current_padded_movie,
                trend_movie=current_trend_movie,
                dt=0, dx=0, dy=0,
                normalize=False)
            current_detrended_std_xy = current_detrended_var_xy.sqrt() / input_movie_std_scale

            self.features.feature_array_list.append(crop_center(
                upsample_to_numpy(current_detrended_std_xy, depth),
                target_width=original_width,
                target_height=original_height)) 
            self.features.feature_depth_list.append(depth)
            self.features.feature_name_list.append(f'detrended_std_{depth}')

            # calculate trend std
            current_trend_var_xy = calculate_cross(
                padded_movie=current_trend_movie,
                trend_movie=None,
                dt=0, dx=0, dy=0,
                normalize=False)
            current_trend_std_xy = current_trend_var_xy.sqrt() / input_movie_std_scale

            self.features.feature_array_list.append(crop_center(
                upsample_to_numpy(current_trend_std_xy, depth),
                target_width=original_width,
                target_height=original_height))
            self.features.feature_depth_list.append(depth)
            self.features.feature_name_list.append(f'trend_std_{depth}')

            # calculate trend mean
            current_trend_mean_xy = unpad_frame(
                current_trend_movie.padded_movie_txy.mean(0),
                current_trend_movie) / input_movie_std_scale

            self.features.feature_array_list.append(crop_center(
                upsample_to_numpy(current_trend_mean_xy, depth),
                target_width=original_width,
                target_height=original_height))
            self.features.feature_depth_list.append(depth)
            self.features.feature_name_list.append(f'trend_mean_{depth}')

            for (dt, dx, dy) in corr_displacement_list:

                logging.debug(f'Calculating x-corr ({dt}, {dx}, {dy}) at depth {depth} for detrended movie...')
                current_cross_corr_xy = calculate_cross(
                    padded_movie=current_padded_movie,
                    trend_movie=current_trend_movie,
                    dt=dt, dx=dx, dy=dy,
                    normalize=False)
                # normed_current_cross_corr_xy = current_cross_corr_xy
                normed_current_cross_corr_xy = current_cross_corr_xy / (const.EPS + current_detrended_var_xy)

                self.features.feature_array_list.append(crop_center(
                    upsample_to_numpy(normed_current_cross_corr_xy, depth),
                    target_width=original_width,
                    target_height=original_height))
                self.features.feature_depth_list.append(depth)
                self.features.feature_name_list.append(f'detrended_corr_{depth}_{dt}_{dx}_{dy}')

                logging.debug(f'Calculating x-corr ({dt}, {dx}, {dy}) at depth {depth} for the trend...')
                current_cross_corr_xy = calculate_cross(
                    padded_movie=current_trend_movie,
                    trend_movie=None,
                    dt=dt, dx=dx, dy=dy,
                    normalize=False)
                # normed_current_cross_corr_xy = current_cross_corr_xy
                normed_current_cross_corr_xy = current_cross_corr_xy / (const.EPS + current_trend_var_xy)

                self.features.feature_array_list.append(crop_center(
                    upsample_to_numpy(normed_current_cross_corr_xy, depth),
                    target_width=original_width,
                    target_height=original_height))
                self.features.feature_depth_list.append(depth)
                self.features.feature_name_list.append(f'trend_corr_{depth}_{dt}_{dx}_{dy}')
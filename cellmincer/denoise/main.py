import os
import logging
import pprint
import time

import json
import pickle
import tifffile

from skvideo import io as skio
from matplotlib.colors import Normalize
import matplotlib.pylab as plt
import numpy as np
import torch
from typing import List, Optional

from cellmincer.containers import Noise2Self
from cellmincer.util import crop_center
    
class Denoise:
    def __init__(
            self,
            input_dir: str,
            output_dir: str,
            model_state: str,
            config: dict,
            avi_enabled: bool,
            avi_frames: Optional[List[int]],
            avi_sigma: Optional[List[int]]):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
        ws_denoising_list, self.denoising_model = Noise2Self(
            datasets=[input_dir],
            config=config).get_resources()
        self.ws_denoising = ws_denoising_list[0]
        self.denoising_model.load_state_dict(torch.load(model_state))

        self.avi_enabled = avi_enabled
        self.avi_frames = avi_frames if avi_frames is not None else [0, self.ws_denoising.n_frames]
        self.avi_sigma = avi_sigma if avi_sigma is not None else [0, 10]
    
    def run(self):
        logging.info('Denoising movie...')
        self.denoising_model.eval()

        denoised_txy = self.denoising_model.denoise_movie(self.ws_denoising).numpy()

        if self.avi_enabled:
            denoised_norm_txy = self.normalize_movie(
                denoised_txy,
                sigma_lo=self.avi_sigma[0],
                sigma_hi=self.avi_sigma[1])

            writer = skio.FFmpegWriter(
                os.path.join(self.output_dir, f'denoised.avi'),
                outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
            
            i_start, i_end = self.avi_frames
            if i_start < 0:
                i_start += self.ws_denoising.n_frames
            if i_end < 0:
                i_end += self.ws_denoising.n_frames
            
            logging.info(f'Writing .avi with sigma range={self.avi_sigma}; frames=[{i_start}, {i_end}]')

            for i_frame in range(i_start, i_end):
                writer.writeFrame(denoised_norm_txy[i_frame].T[None, ...])
            writer.close()

        denoised_txy *= self.ws_denoising.cached_features.norm_scale
        denoised_txy += self.ws_denoising.bg_movie_txy

        tifffile.imwrite(
            os.path.join(self.output_dir, f'denoised_tyx.tif'),
            denoised_txy.transpose((0, 2, 1)))
        logging.info('Denoising done.')

    def normalize_movie(
            self,
            movie_txy: np.ndarray,
            sigma_lo: float,
            sigma_hi: float,
            mean=None,
            std=None,
            max_intensity=255):
        if mean is None:
            mean = movie_txy.mean()
        if std is None:
            std = movie_txy.std()
        z_movie_txy  = (movie_txy - mean) / std
        norm = Normalize(vmin=sigma_lo, vmax=sigma_hi, clip=True)
        return max_intensity * norm(z_movie_txy)

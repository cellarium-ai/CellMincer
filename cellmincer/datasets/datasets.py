import os
import logging
import pprint
import time

import json
import pickle

import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from cellmincer.models import \
    DenoisingModel, \
    init_model, \
    get_temporal_order_from_config, \
    get_window_padding_from_config

from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    const

import warnings

# suppress DataLoader pthreadpool warning
warnings.filterwarnings('ignore')

def movie_collate_fn(data):
    return {
        'padded_diff': torch.cat([item['padded_diff'] for item in data], dim=0),
        'features': torch.cat([item['features'] for item in data], dim=0),
        'trend_mean_feature_index': data[0]['trend_mean_feature_index'],
        'detrended_std_feature_index': data[0]['detrended_std_feature_index'],
        'over_pivot': np.array([item['over_pivot'] for item in data])}

class MovieDataset(Dataset):
    def __init__(
            self,
            ws_denoising_list: List[OptopatchDenoisingWorkspace],
            x_window: int,
            y_window: int,
            x_padding: int,
            y_padding: int,
            t_total: int,
            length: int,
            oversample: Optional[dict] = None):
        self.ws_denoising_list = ws_denoising_list
        self.x_window = x_window
        self.y_window = y_window
        self.x_padding = x_padding
        self.y_padding = y_padding
        self.t_total = t_total
        self.length = length
        
        if oversample is None:
            self.pivot = None
        else:
            self.pivot = []
            for ws_denoising in self.ws_denoising_list:
                intensity = []

                for _ in range(oversample['n_sample']):
                    n_frames, width, height = ws_denoising.shape
                    t0 = np.random.randint(n_frames)
                    x0 = np.random.randint(width - self.x_window + 1)
                    y0 = np.random.randint(height - self.y_window + 1)

                    crop_xy = ws_denoising.get_movie_slice(
                        include_bg=False,
                        t_begin=t0,
                        t_length=1,
                        x0=x0,
                        y0=y0,
                        x_window=self.x_window,
                        y_window=self.y_window,
                        x_padding=0,
                        y_padding=0)['diff'].squeeze()
                    intensity.append(crop_xy.mean())

                intensity.sort()
                ds_pivot = intensity[-int(oversample['n_sample'] * oversample['pivot'])]
                self.pivot.append(ds_pivot)

    def __len__(self):
        return self.length
        
    def __getitem__(self, item):
        movie_idx = item % len(self.ws_denoising_list)

        # ignored in sampling if oversampling is not in effect
        over_pivot = item * 2 < self.length if self.pivot is not None else None
        return self._generate_random_slice(movie_idx, over_pivot)

    def _generate_random_slice(self, movie_idx, over_pivot):
        ws_denoising = self.ws_denoising_list[movie_idx]

        n_frames, width, height = ws_denoising.shape

        while True:
            t0 = np.random.randint(n_frames - self.t_total + 1)
            x0 = np.random.randint(width - self.x_window + 1)
            y0 = np.random.randint(height - self.y_window + 1)
            
            crop_xy = ws_denoising.get_movie_slice(
                include_bg=False,
                t_begin=t0 + self.t_total // 2,
                t_length=1,
                x0=x0,
                y0=y0,
                x_window=self.x_window,
                y_window=self.y_window,
                x_padding=0,
                y_padding=0)['diff'].squeeze()
            
            if over_pivot is None or over_pivot ^ (crop_xy.mean() < self.pivot[movie_idx]):
                break

        diff_movie_slice_1txy = ws_denoising.get_movie_slice(
            include_bg=False,
            t_begin=t0,
            t_length=self.t_total,
            x0=x0,
            y0=y0,
            x_window=self.x_window,
            y_window=self.y_window,
            x_padding=self.x_padding,
            y_padding=self.y_padding)['diff']
        feature_slice_list_1fxy = ws_denoising.get_feature_slice(
            x0=x0,
            y0=y0,
            x_window=self.x_window,
            y_window=self.y_window,
            x_padding=self.x_padding,
            y_padding=self.y_padding)
        trend_mean_feature_index = ws_denoising.cached_features.get_feature_index('trend_mean_0')
        detrended_std_feature_index = ws_denoising.cached_features.get_feature_index('detrended_std_0')

        return {
            'padded_diff': diff_movie_slice_1txy,
            'features': feature_slice_list_1fxy,
            'trend_mean_feature_index': trend_mean_feature_index,
            'detrended_std_feature_index': detrended_std_feature_index,
            'over_pivot': over_pivot}

class MovieDataModule(LightningDataModule):
    def __init__(
            self,
            ws_denoising_list: List[OptopatchDenoisingWorkspace],
            x_window: int,
            y_window: int,
            x_padding: int,
            y_padding: int,
            t_total: int,
            n_batch: int,
            length: int,
            oversample: Optional[dict] = None):
        self.n_batch = n_batch
        self.dataset = MovieDataset(
            ws_denoising_list=ws_denoising_list,
            x_window=x_window,
            y_window=y_window,
            x_padding=x_padding,
            y_padding=y_padding,
            t_total=t_total,
            length=length,
            oversample=oversample)
        self.n_global_features = ws_denoising_list[0].n_global_features

    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> "torch.dataloader":
        return DataLoader(
            dataset=self.dataset,
            collate_fn=movie_collate_fn,
            batch_size=self.n_batch,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            shuffle=False)

    def val_dataloader(self) -> "torch.dataloader":
        pass

def build_ws_denoising(
        dataset: str,
        model_config: dict,
        use_memmap: bool,
        device: Optional[torch.device] = None):
    movie_diff = np.load(os.path.join(dataset, 'trend_subtracted.npy'))
    movie_bg_path = os.path.join(dataset, 'trend.npy')

    opto_noise_params_path = os.path.join(dataset, 'noise_params.json')
    with open(opto_noise_params_path, 'r') as f:
        noise_params = json.load(f)

    opto_feature_path = os.path.join(dataset, 'features.pkl')
    with open(opto_feature_path, 'rb') as f:
        feature_container = pickle.Unpickler(f).load()

    padding = max(get_window_padding_from_config(
        model_config=model_config,
        output_min_size=np.arange(1, max(movie_diff.shape[-2:]) + 1)))
    
    return OptopatchDenoisingWorkspace(
        movie_diff=movie_diff,
        movie_bg_path=movie_bg_path,
        noise_params=noise_params,
        features=feature_container,
        x_padding=padding,
        y_padding=padding,
        use_memmap=use_memmap,
        clip=model_config.get('clip', 0),
        padding_mode=model_config['padding_mode'],
        occlude_padding=model_config['occlude_padding'],
        device=device)

def build_datamodule(
        datasets: List[str],
        model_config: dict,
        train_config: dict,
        gpus: int,
        use_memmap: bool) -> MovieDataModule:
    logging.info('Loading datasets...')
    assert all([os.path.exists(dataset) for dataset in datasets])

    ws_denoising_list = []
    for i_dataset, dataset in enumerate(datasets):
        logging.info(f'({i_dataset + 1}/{len(datasets)}) {dataset}')
        ws_denoising_list.append(build_ws_denoising(dataset, model_config, use_memmap))

    return MovieDataModule(
        ws_denoising_list,
        x_window=train_config['x_window'],
        y_window=train_config['y_window'],
        x_padding=train_config['x_padding'],
        y_padding=train_config['y_padding'],
        t_total=get_temporal_order_from_config(model_config) + train_config['t_tandem'] - 1,
        n_batch=train_config['n_batch'],
        length=gpus * train_config['n_batch'],
        oversample=train_config.get('oversample', None))

def wipe_temp_files(movie_dm: MovieDataModule):
    for ws_denoising in movie_dm.dataset.ws_denoising_list:
        for tfile in ws_denoising.tempfiles:
            try:
                os.remove(tfile)
            except FileNotFoundError:
                pass

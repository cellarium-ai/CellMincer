import os
import logging
import pprint
import time

import json
import pickle

import numpy as np
import torch
from typing import List, Tuple, Optional

from cellmincer.models import DenoisingModel, init_model, get_window_padding_from_config
from cellmincer.util import \
    OptopatchBaseWorkspace, \
    OptopatchDenoisingWorkspace, \
    const


class Noise2Self:
    def __init__(
            self,
            datasets: List[str],
            config: dict):
        self.datasets = datasets
        
        self.model_config = config['model']
        self.device = torch.device(config['device'])
        
    def load_datasets(self) -> List[OptopatchDenoisingWorkspace]:
        logging.info('Loading datasets...')
        assert all([os.path.exists(dataset) for dataset in self.datasets])

        ws_denoising_list = []
        for i_dataset, dataset in enumerate(self.datasets):
            logging.info(f'({i_dataset + 1}/{len(self.datasets)}) {dataset}')
            
            base_diff_path = os.path.join(dataset, 'trend_subtracted.npy')
            ws_base_diff = OptopatchBaseWorkspace.from_npy(base_diff_path)

            base_bg_path = os.path.join(dataset, 'trend.npy')
            ws_base_bg = OptopatchBaseWorkspace.from_npy(base_bg_path)

            opto_noise_params_path = os.path.join(dataset, 'noise_params.json')
            with open(opto_noise_params_path, 'r') as f:
                noise_params = json.load(f)

            opto_feature_path = os.path.join(dataset, 'features.pkl')
            with open(opto_feature_path, 'rb') as f:
                feature_container = pickle.Unpickler(f).load()

            padding = max(get_window_padding_from_config(
                model_config=self.model_config,
                output_min_size=np.arange(1, max(ws_base_diff.width, ws_base_diff.height) + 1)))

            ws_denoising_list.append(
                OptopatchDenoisingWorkspace(
                    ws_base_diff=ws_base_diff,
                    ws_base_bg=ws_base_bg,
                    noise_params=noise_params,
                    features=feature_container,
                    x_padding=padding,
                    y_padding=padding,
                    padding_mode=self.model_config['padding_mode'],
                    occlude_padding=self.model_config['occlude_padding'],
                    device=self.device,
                    dtype=const.DEFAULT_DTYPE
                )
            )

        return ws_denoising_list
    
    def instance_model(
        self,
        n_global_features: int) -> DenoisingModel:
    
        self.model_config['n_global_features'] = n_global_features

        denoising_model = init_model(
            self.model_config,
            device=self.device,
            dtype=const.DEFAULT_DTYPE)

        return denoising_model

    def get_resources(self) -> Tuple[List[OptopatchDenoisingWorkspace], DenoisingModel]:
        ws_denoising_list = self.load_datasets()

        denoising_model = self.instance_model(
            n_global_features=ws_denoising_list[0].n_global_features)
        
        return ws_denoising_list, denoising_model

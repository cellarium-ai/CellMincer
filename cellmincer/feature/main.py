import logging
import pprint
import time

import pickle

import numpy as np
import torch

from cellmincer.util import OptopatchBaseWorkspace, OptopatchGlobalFeatureExtractor

class Feature:
    def __init__(
            self,
            input_file: str,
            output_file: str):
        
        self.ws_base = OptopatchBaseWorkspace.from_npy(input_file)
        self.output_file = output_file
        self.device = torch.device('cpu')

    def run(self):
        feature_extractor = OptopatchGlobalFeatureExtractor(
            ws_base=self.ws_base,
            max_depth=1,
            device=self.device)

        with open(self.output_file, 'wb') as f:
            pickle.dump(feature_extractor.features, f)

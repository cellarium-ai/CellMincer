import os
import logging

import numpy as np
from typing import List, Optional

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from cellmincer.datasets import build_datamodule, wipe_temp_files

from cellmincer.models import \
    init_model, \
    get_window_padding_from_config, \
    load_model_from_checkpoint
    
class Train:
    def __init__(
            self,
            datasets: List[str],
            output_dir: str,
            config: dict,
            gpus: int,
            use_memmap: bool,
            pretrain: Optional[str] = None,
            resume: Optional[str] = None,
            checkpoint: Optional[str] = None):
        
        self.model = None
        
        # if finetuning trained model
        if pretrain:
            try:
                self.model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=pretrain)
                logging.info('Loaded pretrained model state.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring pretrained state.')
        
        # if resuming training from saved checkpoint
        self.ckpt_path = None
        if resume is not None and os.path.exists(resume):
            try:
                self.model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=resume)
                self.ckpt_path = resume
                logging.info('Successfully loaded restarted checkpoint.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring resume state.')
                pass
        
        # if resuming training from preempted checkpoint
        if checkpoint is not None and os.path.exists(checkpoint):
            try:
                self.model = load_model_from_checkpoint(
                    model_type=config['model']['type'],
                    ckpt_path=checkpoint)
                self.ckpt_path = checkpoint
                logging.info('Successfully loaded checkpoint.')
            except EOFError:
                logging.warning('Bad checkpoint; ignoring checkpointed state. Expected behavior when first initialized.')
                pass
        
        if self.model:
            train_config = self.model.hparams.train_config
        else:
            train_config = config['train']
            
            # compute training padding with maximal output/input receptive field ratio
            output_min_size = np.arange(config['train']['output_min_size_lo'], config['train']['output_min_size_hi'] + 1)
            output_min_size = output_min_size[(output_min_size % 2) == 0]
            train_padding = get_window_padding_from_config(
                model_config=config['model'],
                output_min_size=output_min_size)
            best_train = np.argmax(output_min_size / (output_min_size + 2 * train_padding))
            train_config['x_window'] = train_config['y_window'] = output_min_size[best_train]
            train_config['x_padding'] = train_config['y_padding'] = train_padding[best_train]
            
        self.movie_dm = build_datamodule(
            datasets=datasets,
            model_config=config['model'],
            train_config=train_config,
            gpus=gpus,
            use_memmap=use_memmap)
        
        if self.model is None:
            logging.info('Initializing new model.')
            model_config = config['model']
            model_config['n_global_features'] = self.movie_dm.n_global_features
            self.model = init_model(
                model_config=model_config,
                train_config=train_config)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.trainer = Trainer(
            strategy=DDPStrategy(find_unused_parameters=True),
            devices=gpus,
            max_epochs=train_config['n_iters'],
            default_root_dir=self.output_dir,
            callbacks=[ModelCheckpoint(dirpath=self.output_dir, save_last=True)],
            sync_batchnorm=True)

    def run(self):
        logging.info('Training model...')
        
        self.trainer.fit(
            model=self.model,
            train_dataloaders=self.movie_dm,
            ckpt_path=self.ckpt_path)

        # save trained model
        logging.info('Training complete.')
        
        # delete any temporary files generated as memory maps
        wipe_temp_files(self.movie_dm)

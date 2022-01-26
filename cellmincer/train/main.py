import os
import logging
import pprint
import time

import json
import pickle
import tarfile

import matplotlib.pylab as plt
import numpy as np
import skimage
import torch
import pandas as pd
from typing import List, Optional, Tuple

from cellmincer.containers import Noise2Self
from cellmincer.models import DenoisingModel, get_window_padding_from_config
from cellmincer.util import \
    const, \
    crop_center, \
    generate_optimizer, \
    generate_lr_scheduler, \
    generate_batch_indices, \
    generate_occluded_training_data, \
    get_noise2self_loss
    
class Train:
    def __init__(
            self,
            inputs: List[str],
            output_dir: str,
            config: dict,
            pretrain: Optional[str] = None,
            checkpoint: Optional[str] = None,
            temp: Optional[str] = None):

        # compute training padding with maximal output/input receptive field ratio
        output_min_size = np.arange(config['train']['output_min_size_lo'], config['train']['output_min_size_hi'] + 1)
        output_min_size = output_min_size[(output_min_size % 2) == 0]
        train_padding = get_window_padding_from_config(
            model_config=config['model'],
            output_min_size=output_min_size)
        best_train = np.argmax(output_min_size / (output_min_size + 2 * train_padding))
        self.x_train_window = self.y_train_window = output_min_size[best_train]
        self.x_train_padding = self.y_train_padding = train_padding[best_train]
        
        self.ws_denoising_list, self.denoising_model = Noise2Self(
            datasets=inputs,
            config=config).get_resources()
        
        self.include_bg = (config['train']['loss_type'] == 'poisson_gaussian')
        
        # log verbose model summary
        logging.info(self.denoising_model.summary(
            ws_denoising=self.ws_denoising_list[0],
            x_window=self.x_train_window,
            y_window=self.y_train_window))
        
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint')):
            os.mkdir(os.path.join(self.output_dir, 'checkpoint'))
        
        self.train_config = config['train']
        self.enable_continuity_reg = self.train_config['enable_continuity_reg']
        self.start_iter = 0
        self.device = torch.device(config['device'])
        
        # initialize with pretrained weights
        if pretrain:
            self.denoising_model.load_state_dict(torch.load(os.path.join(pretrain)))
        
        # initialize optimizer and scheduler
        self.optim = generate_optimizer(
            denoising_model=self.denoising_model,
            optim_params=self.train_config['optim_params'],
            lr=self.train_config['lr_params']['max'])
        self.sched = generate_lr_scheduler(
            optim=self.optim,
            lr_params=self.train_config['lr_params'],
            n_iters=self.train_config['n_iters'])
        
        using_checkpoint = checkpoint is not None and os.path.exists(checkpoint)
        if using_checkpoint:
            logging.info('Attempting to extract checkpoint...')
            try:
                with tarfile.open(checkpoint) as tar:
                    tar.extractall(self.output_dir)
            except tarfile.ReadError:
                logging.info('Checkpoint could not be extracted; reverting to fresh start.')
                using_checkpoint = False
        
        # if continuing from checkpoint
        if using_checkpoint:
            with open(os.path.join(self.output_dir, 'checkpoint/ckpt_index.txt'), 'r') as f:
                self.start_iter = int(f.read()) * self.train_config['checkpoint_every']

            logging.info(f'Restarting training from iteration {self.start_iter}.')
            
            # load training state
            self.denoising_model.load_state_dict(torch.load(
                os.path.join(self.output_dir, 'checkpoint/model.pt')))
            self.optim.load_state_dict(torch.load(
                os.path.join(self.output_dir, 'checkpoint/optim.pt')))
            self.sched.load_state_dict(torch.load(
                os.path.join(self.output_dir, 'checkpoint/sched.pt')))

    def save_checkpoint(self, checkpoint_path: str, index: int):
        # update checkpoint
        with open(os.path.join(self.output_dir, 'checkpoint/ckpt_index.txt'), 'w') as f:
            f.write(str(index))
        
        # write model/training state
        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(self.output_dir, 'checkpoint/model.pt'))
        torch.save(
            self.optim.state_dict(),
            os.path.join(self.output_dir, 'checkpoint/optim.pt'))
        torch.save(
            self.sched.state_dict(),
            os.path.join(self.output_dir, 'checkpoint/sched.pt'))
        
        # tarball checkpoint
        checkpoint_path_tmp = os.path.join(self.output_dir, 'checkpoint_tmp.tar.gz')
        with tarfile.open(checkpoint_path_tmp, 'w:gz') as tar:
            tar.add(os.path.join(self.output_dir, 'checkpoint'), arcname='checkpoint')
        os.replace(checkpoint_path_tmp, checkpoint_path)

    def save_final(self):
        torch.save(
            self.denoising_model.state_dict(),
            os.path.join(self.output_dir, f'model.pt'))

    def run(self):
        logging.info('Training model...')
        
        # select validation frames and shape into batches
        assert self.train_config['n_frames_validation'] % self.train_config['n_batch_validation'] == 0
        
        val_dataset_indices, val_frame_indices = generate_batch_indices(
            self.ws_denoising_list,
            n_batch=self.train_config['n_frames_validation'],
            t_mid=self.denoising_model.t_order,
            dataset_selection='balanced')
        val_batch_shape = (
            self.train_config['n_frames_validation'] // self.train_config['n_batch_validation'],
            self.train_config['n_batch_validation'])
        val_dataset_indices = val_dataset_indices.reshape(val_batch_shape)
        val_frame_indices = val_frame_indices.reshape(val_batch_shape)
        
        last_train_loss = None
        last_val_loss = None

        update_time = True
        for i_iter in range(self.start_iter, self.train_config['n_iters']):
            if update_time:
                start = time.time()
                update_time = False

            norm_p = self.train_config['norm_p']
            # anneal L0 loss
            if norm_p == 0:
                norm_p = (self.train_config['n_iters'] - i_iter) / self.train_config['n_iters']

            c_total_loss_hist = []
            c_rec_loss_hist = []
            c_reg_loss_hist = []

            self.denoising_model.train()
            self.optim.zero_grad()
            
            # aggregate gradients
            for i_loop in range(self.train_config['n_loop']):
                batch_data = generate_occluded_training_data(
                    ws_denoising_list=self.ws_denoising_list,
                    t_order=self.denoising_model.t_order,
                    t_tandem=self.train_config['t_tandem'],
                    n_batch=self.train_config['n_batch_per_loop'],
                    x_window=self.x_train_window,
                    y_window=self.y_train_window,
                    x_padding=self.x_train_padding,
                    y_padding=self.y_train_padding,
                    include_bg=self.include_bg,
                    occlusion_prob=self.train_config['occlusion_prob'],
                    occlusion_radius=self.train_config['occlusion_radius'],
                    occlusion_strategy=self.train_config['occlusion_strategy'],
                    device=self.device,
                    dtype=const.DEFAULT_DTYPE)

                loss_dict = get_noise2self_loss(
                    batch_data=batch_data,
                    ws_denoising_list=self.ws_denoising_list,
                    denoising_model=self.denoising_model,
                    norm_p=norm_p,
                    loss_type=self.train_config['loss_type'],
                    enable_continuity_reg=self.enable_continuity_reg,
                    reg_func=self.train_config['reg_func'],
                    continuity_reg_strength=self.train_config['continuity_reg_strength'],
                    noise_threshold_to_std=self.train_config['noise_threshold_to_std'])

                # calculate gradient
                if self.enable_continuity_reg:
                    total_loss = (loss_dict['rec_loss'] + loss_dict['reg_loss']) / self.train_config['n_loop']
                else:
                    total_loss = loss_dict['rec_loss'] / self.train_config['n_loop']

                total_loss.backward()
                
                c_total_loss_hist.append(total_loss.item() * self.train_config['n_loop'])
                c_rec_loss_hist.append(loss_dict['rec_loss'].item())
                if self.enable_continuity_reg:
                    c_reg_loss_hist.append(loss_dict['reg_loss'].item())

            current_lr = self.sched.get_lr()[0]

            self.optim.step()
            self.sched.step()
            
            last_train_loss = np.mean(c_total_loss_hist)

            # validate with n2s loss on select frames
            if (i_iter + 1) % self.train_config['validate_every'] == 0:
                self.denoising_model.eval()
                
                x_window_full = self.ws_denoising_list[0].width
                y_window_full = self.ws_denoising_list[0].height
                x_padding_full, y_padding_full = self.denoising_model.get_window_padding([x_window_full, y_window_full])
                
                c_val_loss = []
                for val_dataset_batch, val_frame_batch in zip(val_dataset_indices, val_frame_indices):
                    batch_data = generate_occluded_training_data(
                        ws_denoising_list=self.ws_denoising_list,
                        t_order=self.denoising_model.t_order,
                        t_tandem=0,
                        n_batch=self.train_config['n_batch_validation'],
                        # TODO assumes all datasets are compatible with the same full window
                        x_window=x_window_full,
                        y_window=y_window_full,
                        x_padding=x_padding_full,
                        y_padding=y_padding_full,
                        include_bg=self.include_bg,
                        occlusion_prob=1,
                        occlusion_radius=0,
                        occlusion_strategy='validation',
                        dataset_indices=val_dataset_batch,
                        frame_indices=val_frame_batch,
                        device=self.device,
                        dtype=const.DEFAULT_DTYPE)

                    with torch.no_grad():
                        loss_dict = get_noise2self_loss(
                            batch_data=batch_data,
                            ws_denoising_list=self.ws_denoising_list,
                            denoising_model=self.denoising_model,
                            norm_p=norm_p,
                            loss_type=self.train_config['loss_type'],
                            enable_continuity_reg=self.enable_continuity_reg,
                            reg_func=self.train_config['reg_func'],
                            continuity_reg_strength=self.train_config['continuity_reg_strength'],
                            noise_threshold_to_std=self.train_config['noise_threshold_to_std'])
                
                    if self.enable_continuity_reg:
                        c_val_loss.append((loss_dict['rec_loss'] + loss_dict['reg_loss']).item())
                        
                    else:
                        c_val_loss.append(loss_dict['rec_loss'].item())

                last_val_loss = np.mean(c_rec_loss_hist)

            # log training status
            if (i_iter + 1) % self.train_config['log_every'] == 0:
                elapsed = time.time() - start
                update_time = True
                val_loss_str = f'{last_val_loss:.4f}' if last_val_loss is not None else 'n/a'
                logging.info(
                    f'iter {i_iter + 1}/{self.train_config["n_iters"]} | '
                    f'train loss={last_train_loss:.4f} | '
                    f'val loss={val_loss_str} | '
                    f'{self.train_config["log_every"] / elapsed:.2f} iter/s')

            # write checkpoint
            if (i_iter + 1) % self.train_config['checkpoint_every'] == 0:
                if 'checkpoint_path' in self.train_config:
                    index = (i_iter + 1) // self.train_config['checkpoint_every']
                    logging.info(f'Updating and gzipping checkpoint at index {index}')
                    self.save_checkpoint(self.train_config['checkpoint_path'], index)
                else:
                    logging.info(f'No checkpoint path specified; skipping.')

        # save trained model
        logging.info('Training complete; saving model...')
        self.save_final()

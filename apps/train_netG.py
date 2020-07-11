# Usage: 
# python train_net.py -cfg ../configs/example.yaml -- learning_rate 1.0

import sys
import os
import argparse
import time
import cv2
import numpy as np
import glob
import tqdm
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import vtkplotter as vtk

sys.path.insert(0, '../')
from monoport.lib.common.trainer import Trainer
from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.modeling.MonoPortNet import MonoPortNet
from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset
from monoport.lib.dataset.ppl_static import PPLStaticDataset

########################################
## load configs
########################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', default=None, type=str, 
    help='path of the yaml config file')
parser.add_argument(
    '-d', '--dataset', default='static', type=str, 
    choices=['static', 'dynamic'],
    help='on which dataset to do training')
    
argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)
opts = sys.argv[sys.argv.index('--') + 1:]

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(opts)
cfg.freeze()


def train(device='cuda'):
    # setup net 
    net = MonoPortNet(cfg.netG).to(device)

    # setup trainer
    trainer = Trainer(net, cfg, use_tb=True)
    
    # load ckpt
    if os.path.exists(cfg.netG.ckpt_path):
        if 'checkpoints' in cfg.netG.ckpt_path:
            trainer.load_ckpt(cfg.netG.ckpt_path)
        else:
            net.load_legacy_pifu(cfg.netG.ckpt_path)
    else:
        trainer.logger.info(f'ckpt {cfg.ckpt_path} not found.')

    # set dataset
    manager = mp.Manager()
    shared_dict = manager.dict()
    if args.dataset == 'static':
        train_dataset = PPLStaticDataset(
            cfg.dataset, 
            mean=cfg.netG.mean, 
            std=cfg.netG.std,
            training=True,
            split='train')
    elif args.dataset == 'dynamic':
        train_dataset = PPLDynamicDataset(
            cfg.dataset, 
            mean=cfg.netG.mean, 
            std=cfg.netG.std,
            training=True,
            split='train')
    else:
        raise NotImplementedError

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_threads, pin_memory=False, drop_last=True)
    trainer.logger.info(
        f'train data size: {len(train_dataset)}; '+
        f'loader size: {len(train_data_loader)};')
    
    start_iter = trainer.iteration
    start_epoch = trainer.epoch
    # start training
    for epoch in range(start_epoch, cfg.num_epoch):
        trainer.net.train()
            
        loader = iter(train_data_loader)
        niter = len(train_data_loader)        
        
        epoch_start_time = iter_start_time = time.time()
        for iteration in range(start_iter, niter):
            data = next(loader)         
               
            iter_data_time = time.time() - iter_start_time
            global_step = epoch * niter + iteration
            
            # retrieve the data
            image_tensor = data['image'].to(device).float()
            calib_tensor = data['calib'].to(device).float()
            sample_tensor = data['samples_geo'].to(device).float()
            label_tensor = data['labels_geo'].to(device).float()
            sample_tensor = sample_tensor.permute(0, 2, 1) #[bz, 3, N]
            label_tensor = label_tensor.unsqueeze(1) #[bz, 1, N]
            
            # forward
            pred, loss = trainer.net(
                image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
            loss = loss.mean()

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            iter_time = time.time() - iter_start_time
            eta = (niter-iteration) * (time.time()-epoch_start_time) / (iteration-start_iter+1) 

            # print
            if iteration % cfg.freq_plot == 0 and iteration > 0:
                trainer.logger.info(
                    f'Name: {cfg.name}|Epoch: {epoch:02d}({iteration:05d}/{niter})|' \
                    +f'dataT: {(iter_data_time):.3f}|' \
                    +f'totalT: {(iter_time):.3f}|'
                    +f'ETA: {int(eta // 60):02d}:{int(eta - 60 * (eta // 60)):02d}|' \
                    +f'Err:{loss.item():.5f}|'
                )
                trainer.tb_writer.add_scalar('data/loss', loss.item(), global_step)

            # save
            if iteration % cfg.freq_save == 0 and iteration > 0:
                trainer.update_ckpt(
                    f'ckpt_{epoch}.pth', epoch, iteration)

            # # evaluation
            # if iteration % cfg.freq_eval == 0 and iteration > 0:
            #     os.makedirs(
            #         os.path.join(trainer.results_path, 'test'), exist_ok=True
            #     )
            #     trainer.logger.info('testing start')
            #     trainer.net.eval()
            #     test(
            #         trainer.net.module, 
            #         test_engine, 
            #         os.path.join(trainer.results_path, 'test'))
            #     trainer.net.train()

            # # vis
            # if iteration % cfg.freq_vis == 0 and iteration > 0:
            #     os.makedirs(
            #         os.path.join(trainer.results_path, 'test'), exist_ok=True
            #     )
            #     trainer.logger.info('visualize')
            #     trainer.net.eval()
            #     renders = []
            #     for tensor in image_tensor[0:4]:
            #         render = test_engine(
            #             trainer.net.module, image_tensor=tensor.unsqueeze(0))
            #         renders.append(render)
            #     render = np.vstack(renders)
            #     cv2.imwrite(
            #         os.path.join(trainer.results_path, 'test', 'trainset.jpg'), 
            #         render[:, :, ::-1])

            #     renders = []
            #     for image_path in [
            #         '../data/test/0.png', '../data/test/95.png',
            #         '../data/test/143.png', '../data/test/258.png']:
            #         render = test_engine(
            #             trainer.net.module, image_path=image_path)
            #         renders.append(render)
            #     render = np.vstack(renders)
            #     cv2.imwrite(
            #         os.path.join(trainer.results_path, 'test', 'testset.jpg'), 
            #         render[:, :, ::-1])
            #     trainer.net.train()

            # end
            iter_start_time = time.time()
        
        trainer.scheduler.step()
        start_iter = 0


if __name__ == '__main__':
    train()
import os
import sys
from easydict import EasyDict as edict

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from .logger import colorlogger


class Trainer():
    def __init__(self, net, opt=None, use_tb=True):
        self.opt = opt if opt is not None else Trainer.get_default_opt()
        self.net = nn.DataParallel(net)
        self.net.train()

        # set cache path
        self.checkpoints_path = os.path.join(opt.checkpoints_path, opt.name)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.results_path = os.path.join(opt.results_path, opt.name)
        os.makedirs(self.results_path, exist_ok=True)
        
        # set logger
        self.logger = colorlogger(logdir=self.results_path)
        self.logger.info(self.opt)
        
        # set tensorboard
        if use_tb:
            self.tb_writer = SummaryWriter(logdir=self.results_path)

        # set optimizer
        learning_rate = opt.learning_rate
        weight_decay = opt.weight_decay
        momentum = opt.momentum
        if opt.optim == "Adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.net.parameters(), lr=learning_rate, 
                weight_decay=weight_decay)
        elif opt.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=weight_decay)
        elif opt.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=learning_rate)
        elif opt.optim == "RMSprop":
            self.optimizer = torch.optim.RMSprop(
                self.net.parameters(), lr=learning_rate, 
                weight_decay=weight_decay, momentum=momentum)
        else:
            raise NotImplementedError
        
        # set scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=opt.schedule, gamma=opt.gamma)
        
        self.epoch = 0
        self.iteration = 0

    def update_ckpt(self, filename, epoch, iteration, **kwargs):
        # `kwargs` can be used to store loss, accuracy, epoch, iteration and so on.
        ckpt_path = os.path.join(self.checkpoints_path, filename)
        saved_dict = {
            "opt": self.opt,
            "net": self.net.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
        }
        for k, v in kwargs.items():
            saved_dict[k] = v
        torch.save(saved_dict, ckpt_path)
        self.logger.info(f'save ckpt to {ckpt_path}')

    def load_ckpt(self, ckpt_path):
        self.logger.info(f'load ckpt from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.net.module.load_state_dict(ckpt["net"])

        if self.opt.resume:
            self.logger.info('loading for optimizer & scheduler ...')
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            
            self.epoch = ckpt["epoch"]
            self.logger.info(f'loading for start epoch ... {self.epoch}')
            self.iteration = ckpt["iteration"]
            self.logger.info(f'loading for start iteration ... {self.iteration}')

    @classmethod
    def get_default_opt(cls):
        opt = edict()

        opt.name = 'example'
        opt.checkpoints_path = '../data/checkpoints/'
        opt.results_path = '../data/results/'
        opt.learning_rate = 1e-3
        opt.weight_decay = 0.0
        opt.momentum = 0.0
        opt.optim = 'RMSprop'
        opt.schedule = [40, 60]
        opt.gamma = 0.1
        opt.resume = False 
        return opt
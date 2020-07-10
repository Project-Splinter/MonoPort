import sys
import os
import argparse
import glob

import torch

from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.modeling.MonoPortNet import MonoPortNet
from monoport.lib.modeling.MonoPortNet import PIFuNetG, PIFuNetC

import streamer_pytorch as streamer
import human_inst_seg

########################################
## load configs
########################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', default=None, type=str, 
    help='path of the yaml config file')
parser.add_argument(
    '--camera', action="store_true")
parser.add_argument(
    '--images', default="", nargs="*")
parser.add_argument(
    '--image_folder', default=None)
parser.add_argument(
    '--videos', default="", nargs="*")
parser.add_argument(
    '--loop', action="store_true")
parser.add_argument(
    '--vis', action="store_true")
parser.add_argument(
    '--use_VRweb', action="store_true")

argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)
opts = sys.argv[sys.argv.index('--') + 1:]

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(opts)
cfg.freeze()


########################################
## access avaiable GPUs
########################################
device_count = torch.cuda.device_count()
if device_count == 1:
    cuda_backbone_G='cuda:0' 
    cuda_backbone_C='cuda:0'
    cuda_recon='cuda:0'
    cuda_color='cuda:0'
elif device_count == 2:
    cuda_backbone_G='cuda:1' 
    cuda_backbone_C='cuda:1'
    cuda_recon='cuda:0'
    cuda_color='cuda:1'
else:
    raise NotImplementedError
    

########################################
## load networks
########################################
print (f'loading networkG from {cfg.netG.ckpt_path} ...')
netG = MonoPortNet(cfg.netG)
assert os.path.exists(cfg.netG.ckpt_path), 'we need a ckpt to run RTL demo.'
netG.load_legacy_pifu(cfg.netG.ckpt_path)

netG.image_filter = netG.image_filter.to(cuda_backbone_G)
netG.surface_classifier = netG.surface_classifier.to(cuda_recon)

if os.path.exists(cfg.netC.ckpt_path):
    print (f'loading networkC from {cfg.netC.ckpt_path} ...')
    netC = MonoPortNet(cfg.netC)
    netC.load_legacy_pifu(cfg.netC.ckpt_path)

    netC.image_filter = netC.image_filter.to(cuda_backbone_C)
    netC.surface_classifier = netC.surface_classifier.to(cuda_color)
else:
    print (f'we are not loading netC ...')


########################################
## initialize data streamer
########################################
print (f'initialize data streamer ...')
if args.camera:
    data_stream = streamer.CaptureStreamer(
        mean=cfg.netG.mean, std=cfg.netG.std)
elif len(args.videos) > 0:
    data_stream = streamer.VideoListStreamer(
        args.videos * (10 if args.loop else 1),
        mean=cfg.netG.mean, std=cfg.netG.std)
elif len(args.images) > 0:
    data_stream = streamer.ImageListStreamer(
        args.images * (10000 if args.loop else 1),
        mean=cfg.netG.mean, std=cfg.netG.std)
elif args.image_folder is not None:
    images = sorted(glob.glob(args.image_folder+'/*.jpg'))
    images += sorted(glob.glob(args.image_folder+'/*.png'))
    data_stream = streamer.ImageListStreamer(
        images * (10 if args.loop else 1),
        mean=cfg.netG.mean, std=cfg.netG.std)

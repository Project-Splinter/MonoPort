import sys
import os
import argparse
import glob

import torch
import torch.nn.functional as F

from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.modeling.MonoPortNet import MonoPortNet
from monoport.lib.modeling.MonoPortNet import PIFuNetG, PIFuNetC

########################################
## load configs
########################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', default=None, type=str, 
    help='path of the yaml config file')

argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)
opts = sys.argv[sys.argv.index('--') + 1:]

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(opts)
cfg.freeze()


########################################
## load networks
########################################
print (f'loading networkG from {cfg.netG.ckpt_path} ...')
netG = MonoPortNet(cfg.netG)
assert os.path.exists(cfg.netG.ckpt_path), 'we need a ckpt to run RTL demo.'
if 'checkpoints' in cfg.netG.ckpt_path:
    ckpt = torch.load(cfg.netG.ckpt_path, map_location="cpu")
    netG.load_state_dict(ckpt['net'])
else:
    netG.load_legacy_pifu(cfg.netG.ckpt_path)
    

# Input to the model
x = torch.randn(1, 3, 512, 512, requires_grad=True)
torch_out = netG.image_filter(x)

# Export the model
torch.onnx.export(netG.image_filter,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "./data/PIFu/netG_filter.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )


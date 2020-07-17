# torch2trt should be imported before torch
from torch2trt import torch2trt
import torch
from torch import nn
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['pool']

# create some regular pytorch model...
model = resnet_fpn_backbone(
    backbone_name='resnet152', pretrained=False
).cuda().eval()
model = ModelWrapper(model)

# create example data
x = torch.ones((1, 3, 512, 512)).cuda()

model_trt = torch2trt(model, [x])

exit()


resnet152_fpn = resnet_fpn_backbone(backbone_name='resnet152', pretrained=False).eval()
x = torch.rand(1, 3, 512, 512)
outputs = resnet152_fpn(x)
for k, v in outputs.items():
    print (k, v.shape)

exit()

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
x = torch.randn((1, 3, 512, 512)).cuda()
model = netG.image_filter.cuda().eval()

model_trt = torch2trt(model, [x])
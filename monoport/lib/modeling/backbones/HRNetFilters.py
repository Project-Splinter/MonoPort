from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = lambda planes: nn.BatchNorm2d(planes, momentum=0.9)
# BatchNorm2d = lambda planes: nn.GroupNorm(32, planes)
Conv2d = nn.Conv2d

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = Conv2d(
            inplanes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(
            planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        if self.inplanes != self.planes*self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(self.inplanes, self.planes * self.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.inplanes != self.planes*self.expansion:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1,
                            bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if self.inplanes != self.planes*self.expansion:
            self.downsample = nn.Sequential(
                Conv2d(self.inplanes, self.planes * self.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.inplanes != self.planes*self.expansion:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                Conv2d(num_inchannels[j], num_outchannels_conv3x3,
                                       3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HRNet, self).__init__()
        self.cfg = cfg

        blocks_dict = {
            'Basic': BasicBlock,
            'Bottleneck': Bottleneck
        }

        self.blocks_dict = blocks_dict
        self.inplanes = 64

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = cfg["STAGE1"]
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        # stage1_out_channel = block.expansion*num_channels
        # self.layer1 = self._make_layer(Bottleneck, self.inplanes, 64, 4)

        if "STAGE2" in cfg:
            self.stage2_cfg = cfg["STAGE2"]
            num_channels = self.stage2_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage2_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition1 = self._make_transition_layer([256], num_channels)
            self.stage2, pre_stage_channels = self._make_stage(
                self.stage2_cfg, num_channels)

        if "STAGE3" in cfg:
            self.stage3_cfg = cfg["STAGE3"]
            num_channels = self.stage3_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage3_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition2 = self._make_transition_layer(
                pre_stage_channels, num_channels)
            self.stage3, pre_stage_channels = self._make_stage(
                self.stage3_cfg, num_channels)

        if "STAGE4" in cfg:
            self.stage4_cfg = cfg["STAGE4"]
            num_channels = self.stage4_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage4_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))]
            self.transition3 = self._make_transition_layer(
                pre_stage_channels, num_channels)
            self.stage4, pre_stage_channels = self._make_stage(
                self.stage4_cfg, num_channels, multi_scale_output=True)

        if "last_layer" in self.cfg and self.cfg["last_layer"] == True:
            last_inp_channels = int(sum(pre_stage_channels))
            self.last_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=256,
                    kernel_size=1,
                    stride=1,
                    padding=1)
            )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        Conv2d(num_channels_pre_layer[i],
                               num_channels_cur_layer[i],
                               3,
                               1,
                               1,
                               bias=False),
                        BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        layers.append(block(inplanes, planes, stride))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = self.blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        y_list = [x]

        if "STAGE2" in self.cfg:    
            x_list = []
            for i in range(self.stage2_cfg['NUM_BRANCHES']):
                if self.transition1[i] is not None:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(x)
            y_list = self.stage2(x_list)

        if "STAGE3" in self.cfg:
            x_list = []
            for i in range(self.stage3_cfg['NUM_BRANCHES']):
                if self.transition2[i] is not None:
                    x_list.append(self.transition2[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage3(x_list)

        if "STAGE4" in self.cfg:    
            x_list = []
            for i in range(self.stage4_cfg['NUM_BRANCHES']):
                if self.transition3[i] is not None:
                    x_list.append(self.transition3[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            y_list = self.stage4(x_list)

        if "last_layer" in self.cfg and self.cfg["last_layer"] == True:
            y0_h, y0_w = y_list[0].size(2), y_list[0].size(3)
            y1 = F.interpolate(y_list[1], size=(y0_h, y0_w), mode='bilinear', align_corners=True)
            y2 = F.interpolate(y_list[2], size=(y0_h, y0_w), mode='bilinear', align_corners=True)
            y3 = F.interpolate(y_list[3], size=(y0_h, y0_w), mode='bilinear', align_corners=True)
            y = torch.cat([y_list[0], y1, y2, y3], 1)
            y = self.last_layer(y)
            return [(y,)]
        else:
            return [tuple(y_list)]

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def HRNetV2_W18(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_CHANNELS": [64],
            "NUM_BLOCKS": [4],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [18, 36],
            "BLOCK": "Bottleneck",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [18, 36, 72],
            "BLOCK": "Bottleneck",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [18, 36, 72, 144],
            "BLOCK": "Bottleneck",
            "FUSE_METHOD": "SUM",
        },
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [18, 36],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [18, 36, 72],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [18, 36, 72, 144],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2_balance_gn32(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [64, 128],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [64, 128, 256],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [64, 128, 256, 384],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "last_layer": False
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2_balance(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [80, 160],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [80, 160, 240],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [80, 160, 240, 360],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "last_layer": False
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2_balance_last(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [80, 160],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [80, 160, 240],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [80, 160, 240, 360],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "last_layer": True
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2_balance_v2(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [80, 160],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [80, 160, 380],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [80, 160, 380, 520],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "last_layer": False
    }

    model = HRNet(cfg)
    model.init_weights()
    return model

def HRNetV2_W18_small_v2_deeper(pretrained=False):
    cfg = {
        "STAGE1":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "NUM_BLOCKS": [2],
            "NUM_CHANNELS": [64],
            "BLOCK": "Bottleneck",
        },
        "STAGE2":{
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [80, 160],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3":{
            "NUM_MODULES": 2,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [2, 2, 3],
            "NUM_CHANNELS": [80, 160, 320],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4":{
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [2, 2, 3, 4],
            "NUM_CHANNELS": [80, 160, 320, 480],
            "BLOCK": "Basic",
            "FUSE_METHOD": "SUM",
        },
        "last_layer": False
    }

    model = HRNet(cfg)
    model.init_weights()
    return model
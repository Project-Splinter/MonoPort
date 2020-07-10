import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalizer(nn.Module):
    def __init__(self, opt):
        super(DepthNormalizer, self).__init__()
        self.opt = opt

    def forward(self, z, calibs=None, index_feat=None):
        '''
        Normalize z_feature
        :param z_feat: [B, 1, N] depth value for z in the image coordinate system
        :return:
        '''
        if self.opt.soft_onehot:
            soft_dim = self.opt.soft_dim
            
            # [-1, +1] -> [0, soft_dim-1]
            z_feat = torch.zeros(z.size(0), soft_dim, z.size(2)).to(z.device) # [1, 64, 10000]
            z_norm = (z.clamp(-1, 1) + 1) / 2.0 * (soft_dim - 1)
            z_floor = torch.floor(z_norm) #[1, 1, 10000]
            z_ceil = torch.ceil(z_norm)

            z_floor_value = 1 - (z_norm - z_floor) #[1, 1, 10000]
            z_ceil_value = 1 - (z_ceil - z_norm)

            z_feat = z_feat.scatter(dim=1, index=z_floor.long(), src=z_floor_value)
            z_feat = z_feat.scatter(dim=1, index=z_ceil.long(), src=z_ceil_value)
        else:
            z_feat = z * self.opt.scale
        return z_feat


def PIFuNomalizer(*args, **kwargs):
    from yacs.config import CfgNode as CN
    opt = CN()
    opt.soft_onehot = False
    opt.scale =  512 // 2 / 200.0
    return DepthNormalizer(opt)
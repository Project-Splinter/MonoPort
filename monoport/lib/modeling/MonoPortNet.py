import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import index, orthogonal, perspective
from .normalizers import *
from .backbones import *
from .heads import *


class MonoPortNet(nn.Module):
    def __init__(self, opt_net):
        """The geometry network

        Arguments:
            opt_net {edict} -- options for netG/netC
        """         
        super().__init__()
        self.opt = opt_net
        assert opt_net.projection in ['orthogonal', 'perspective']
        
        # modules
        self.image_filter = globals()[opt_net.backbone.IMF](opt_net.backbone)
        self.surface_classifier = globals()[opt_net.head.IMF](opt_net.head)
        
        # operations
        self.projection = globals()[opt_net.projection]
        self.normalizer = globals()[opt_net.normalizer.IMF](opt_net.normalizer)
        

    def filter(self, images, feat_prior=None):
        """Filter the input images

        Arguments:
            images {torch.tensor} -- input images with shape [B, C, H, W]

        Returns:
            list(list(torch.tensor)) -- image feature lists. <multi-stage <multi-level>>
        """
        feats_stages = self.image_filter(images)
        if feat_prior is not None: # for netC
            feats_stages = [
                (torch.cat([feat_prior, feat_per_lvl], dim=1) for feat_per_lvl in feats)
                for feats in feats_stages]
        return feats_stages

    def query(self, feats_stages, points, calibs=None, transforms=None):
        """Given 3D points, query the network predictions for each point.

        Arguments:
            feats_stages {list(list(torch.tensor))} -- image feature lists. First level list 
                is for multi-stage losses. Second level list is for multi-level features.
            points {torch.tensor} -- [B, 3, N] world space coordinates of points.
            calibs {torch.tensor} -- [B, 3, 4] calibration matrices for each image.
        
        Keyword Arguments:
            transforms {torch.tensor} -- Optional [B, 2, 3] image space coordinate transforms

        Returns:
            list(torch.tensor) -- predictions for each point at each stage. list of [B, Res, N].
        """
        if not self.training:
            feats_stages = [feats_stages[-1]]

        if calibs is None:
            xyz = points
        else:
            xyz = self.projection(points, calibs, transforms)

        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)

        z_feat = self.normalizer(z, calibs=calibs)
        pred_stages = []
        for feats in feats_stages: # each stage
            # concatenate feats cross all the levels. [B, Feat_all, N]
            # TODO: another option here is to do ADD op if all the levels
            # have same feature dimentions.
            point_local_feat = torch.cat(
                [index(feat_per_lvl, xy) for feat_per_lvl in feats] + [z_feat], 1)

            # [B, Res, N]. Res=1 here.
            pred = self.surface_classifier(point_local_feat)

            # out of image plane is always set to 0
            preds = in_img[:, None].float() * pred
            pred_stages.append(pred)
        return pred_stages

    def get_loss(self, pred_stages, labels):
        """Calculate loss between predictions and labels

        Arguments:
            pred_stages {list(torch.tensor)} -- predictions at each stage. list of [B, Res, N]
            labels {torch.tensor} -- labels. typically [B, Res, N]

        Raises:
            NotImplementedError:

        Returns:
            torch.tensor -- average loss cross stages.
        """
        if self.opt.loss.IMF == 'MSE':
            loss_func = F.mse_loss
        elif self.opt.loss.IMF == 'L1':
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        
        loss = 0    
        for pred in pred_stages:
            loss += loss_func(pred, labels)
        loss /= len(pred_stages)
        return loss
    

    def forward(self, images, points, calibs, transforms=None, labels=None, feat_prior=None):
        """Forward function given points and calibs

        Arguments:
            images {torch.tensor} -- shape of [B, C, H, W]
            points {torch.tensor} -- shape of [B, 3, N]
            calibs {torch.tesnor} -- shape of [B, 3, 4]

        Keyword Arguments:
            transforms {torch.tensor} -- shape of [B, 2, 3] (default: {None})
            labels {torch.tensor} -- shape of [B, Res, N] (default: {None})

        Returns:
            torch.tensor, [torch.scaler] -- return preds at last stages. shape of [B, Res, N]
        """
        feats_stages = self.filter(images, feat_prior)
        pred_stages = self.query(feats_stages, points, calibs, transforms)
        
        if labels is not None:
            loss = self.get_loss(pred_stages, labels)
            return pred_stages[-1], loss
        else:
            return pred_stages[-1]

    # def load_legacy(self, ckpt_path):
    #     ckpt = torch.load(ckpt_path, map_location="cpu")["netG"]
    #     backbone_dict = {
    #         k.replace("image_filter.", ""): v for k, v in ckpt.items() if "image_filter" in k}
    #     head_dict = {
    #         k.replace("surface_classifier.", ""): v for k, v in ckpt.items() if "surface_classifier" in k}
    #     self.image_filter.load_state_dict(backbone_dict)
    #     self.surface_classifier.load_state_dict(head_dict)

    def load_legacy_pifu(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        backbone_dict = {
            k.replace("image_filter.", ""): v for k, v in ckpt.items() if "image_filter" in k}
        head_dict = {
            k.replace("surface_classifier.conv", "filters."): v for k, v in ckpt.items() if "surface_classifier" in k}
        self.image_filter.load_state_dict(backbone_dict)
        self.surface_classifier.load_state_dict(head_dict)


def PIFuNetG():
    from yacs.config import CfgNode as CN
    opt_net = CN()
    opt_net.projection = "orthogonal" 
    
    # --- netG:backbone options ---
    opt_net.backbone = CN()
    opt_net.backbone.IMF = 'PIFuHGFilters'

    # --- netG:normalizer options ---
    opt_net.normalizer = CN()
    opt_net.normalizer.IMF = 'PIFuNomalizer'
    
    # --- netG:head options ---
    opt_net.head = CN()
    opt_net.head.IMF = 'PIFuNetGMLP'

    # --- netG:loss options ---
    opt_net.loss = CN()
    opt_net.loss.IMF = 'MSE'

    return MonoPortNet(opt_net)


def PIFuNetC():
    from yacs.config import CfgNode as CN
    opt_net = CN()
    opt_net.projection = "orthogonal" 
    
    # --- netG:backbone options ---
    opt_net.backbone = CN()
    opt_net.backbone.IMF = 'PIFuResBlkFilters'

    # --- netG:normalizer options ---
    opt_net.normalizer = CN()
    opt_net.normalizer.IMF = 'PIFuNomalizer'
    
    # --- netG:head options ---
    opt_net.head = CN()
    opt_net.head.IMF = 'PIFuNetCMLP'

    # --- netG:loss options ---
    opt_net.loss = CN()
    opt_net.loss.IMF = 'L1'

    return MonoPortNet(opt_net)

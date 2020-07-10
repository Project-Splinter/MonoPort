import numpy as np
import os
import glob
import torch
import random

from .utils import load_image


def load_calib(calib_path, render_size=512):
    # loading calibration data
    param = np.load(calib_path, allow_pickle=True)
    # pixel unit / world unit
    ortho_ratio = param.item().get('ortho_ratio')
    # world unit / model unit
    scale = param.item().get('scale')
    # camera center world coordinate
    center = param.item().get('center')
    # model rotation
    R = param.item().get('R')

    translate = -np.matmul(R, center).reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    # Match camera space to image pixel space
    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = scale / ortho_ratio
    scale_intrinsic[1, 1] = -scale / ortho_ratio
    scale_intrinsic[2, 2] = scale / ortho_ratio
    # Match image pixel space to image uv space
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(render_size // 2)
    uv_intrinsic[1, 1] = 1.0 / float(render_size // 2)
    uv_intrinsic[2, 2] = 1.0 / float(render_size // 2)

    intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib = torch.from_numpy(calib_mat)
    return calib


class PPLStaticDataset():
    def __init__(self, 
                 cfg, 
                 mean=(0.5, 0.5, 0.5), 
                 std=(0.5, 0.5, 0.5), 
                 training=True, 
                 shared_dict=None):
        self.root = '/home/rui/local/projects/release/PIFu/data/static/'
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.training = training
        self.shared_dict = shared_dict
        self.rotations = range(0, 360, 1)
        self.motion_list = self.get_motion_list()
        
    def __len__(self):
        return len(self.motion_list) * len(self.rotations)

    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print(e)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def get_item(self, index):  
        rid = index % len(self.rotations)
        mid = index // len(self.rotations)

        rotation = self.rotations[rid]
        motion = self.motion_list[mid]

        image_path = self.get_image_path(motion, rotation)
        mask_path = self.get_mask_path(motion, rotation)
        if self.training:
            image, mask = load_image(
                image_path, mask_path, 
                mean=(0.5, 0.5, 0.5), 
                std=(0.5, 0.5, 0.5),
                blur=self.cfg.blur,
                brightness=self.cfg.aug_bri, 
                contrast=self.cfg.aug_con, 
                saturation=self.cfg.aug_sat, 
                hue=self.cfg.aug_hue)
        else:
            image, mask = utils.load_image(image_path, mask_path)

        calib_path = self.get_calib_path(motion, rotation)
        calib = load_calib(calib_path)

        # left-right flip aug
        if self.training and random.random() < 0.5:
            calib[0, :] *= -1
            image = image.flip(dims=[2])
            mask = mask.flip(dims=[2])            

        # return data dict
        data_dict = {
            'motion': str(motion),
            'rotation': rotation,
            'image': image,
            'mask': mask,
            'calib': calib,
        }

        # sampling
        if self.cfg.num_sample_geo:
            samples_geo, labels_geo = self.get_sampling_geo(motion)
            data_dict.update({'samples_geo': samples_geo, 'labels_geo': labels_geo})
        
        if self.cfg.num_sample_color:
            raise NotImplementedError

        return data_dict

    def get_motion_list(self):
        all_subjects = os.listdir(os.path.join(self.root, 'RENDER'))
        val_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(val_subjects) == 0:
            return sorted(all_subjects)
        if self.training:
            return sorted(list(set(all_subjects) - set(val_subjects)))
        else:
            return sorted(list(val_subjects))    

    def get_texture_path(self, motion, rotation):
        return os.path.join(self.root, 'UV_RENDER', motion, f'{rotation}_0_00.jpg')

    def get_image_path(self, motion, rotation):
        return os.path.join(self.root, 'RENDER', motion, f'{rotation}_0_00.jpg')

    def get_mask_path(self, motion, rotation):
        return os.path.join(self.root, 'MASK', motion, f'{rotation}_0_00.png')

    def get_mesh_path(self, motion):
        return os.path.join(self.root, '100k', f'{motion}_100k.obj')

    def get_calib_path(self, motion, rotation):
        return os.path.join(self.root, 'PARAM', motion, f'{rotation}_0_00.npy')
    
    def get_sampling_geo(self, motion):
        cache_files = glob.glob(os.path.join(self.root, 'SAMPLE', motion, "*.pt"))
        cache = torch.load(random.choice(cache_files))
        samples = cache["samples"].float()
        labels = cache["labels"].float()
        return samples, labels


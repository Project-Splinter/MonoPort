import numpy as np
import os
import glob
import torch
import random
import tqdm
import tinyobjloader

from .utils import load_image, projection


def load_calib(calib_path, render_size=512):
    calib_data = np.loadtxt(calib_path, dtype=float)
    extrinsic = calib_data[:4, :4]
    intrinsic = calib_data[4:8, :4] 
    calib_mat = np.matmul(intrinsic, extrinsic)
    calib = torch.from_numpy(calib_mat)
    return calib


def load_obj_verts(mesh_path):
    # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(mesh_path)

    if ret == False:
        print("Failed to load : ", mesh_path)
        return None

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    verts = np.array(attrib.vertices).reshape(-1, 3)
    return verts


class PPLDynamicDataset():
    def __init__(self, 
                 cfg, 
                 mean=(0.5, 0.5, 0.5), 
                 std=(0.5, 0.5, 0.5), 
                 training=True, 
                 split='train',
                 shared_dict=None):
        self.root = '/home/rui/local/projects/MonoPortDataset/data/'
        self.root_render = '/media/linux_data/data/pifu_orth_v1/'
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.training = training if split == 'train' else False
        self.split = split
        self.shared_dict = shared_dict
        self.rotations = range(0, 360, 1)
        self.motion_list = self.get_motion_list()
        self.santity_check()
        
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

        calib_path = self.get_calib_path(motion, rotation)
        calib = load_calib(calib_path)

        # align        
        if self.cfg.align_hip:
            skel_path = self.get_skeleton_path(motion)
            center = np.loadtxt(skel_path, usecols=[1, 2, 3])[1, :] / 100
            center_proj = projection(center.reshape(1, 3), calib).reshape(3,)
            calib[2, 3] -= center_proj[2]
        else:
            center = np.loadtxt(self.get_center_path(motion)).reshape(1, 3)
            center_proj = projection(center, calib).reshape(3,)
            calib[2, 3] -= center_proj[2]
        
        # load image
        image_path = self.get_image_path(motion, rotation)
        if self.training:
            scale = random.uniform(0.9, 1.1)
            calib[0:3] *= scale
            image, mask = load_image(
                image_path, None,
                crop_size=int(512/scale), 
                input_size=512, 
                mean=self.mean, 
                std=self.std,
                blur=self.cfg.blur,
                brightness=self.cfg.aug_bri, 
                contrast=self.cfg.aug_con, 
                saturation=self.cfg.aug_sat, 
                hue=self.cfg.aug_hue)
        else:
            image, mask = load_image(
                image_path, None,
                mean=self.mean, 
                std=self.std)

        # left-right flip aug
        if self.training and random.random() < 0.5:
            calib[0, :] *= -1
            image = image.flip(dims=[2])
            mask = mask.flip(dims=[2])            

        # return data dict
        data_dict = {
            'motion': str(motion),
            'rotation': rotation,
            'image': image.float(),
            'mask': mask.float(),
            'calib': calib.float(),
            'mesh_path': self.get_mesh_path(motion),
        }

        # sampling
        if self.cfg.num_sample_geo:
            samples_geo, labels_geo = self.get_sampling_geo(motion)
            data_dict.update({
                'samples_geo': samples_geo.float(), 
                'labels_geo': labels_geo.float()})
        
        if self.cfg.num_sample_color:
            raise NotImplementedError

        return data_dict

    def get_motion_list(self):
        # val motions
        val_motions = []
        val_subjects = np.loadtxt(os.path.join(self.root, 'renderppl', 'val.txt'), dtype=str)

        if self.cfg.score_filter > 0:
            tags = np.loadtxt(
                './data/dynamic_chamfer.txt', dtype=str, usecols=[0, 1, 2]
                )[::4]
            scores = np.loadtxt(
                './data/dynamic_chamfer.txt', dtype=float, usecols=[4]
                ).reshape(-1, 4).mean(axis=1)
            tags = tags[scores < self.cfg.score_filter]
            train_motions = [
                [subject, action, int(frame)] for (subject, action, frame) in tags]
        else:
            # scan all motions
            paths = sorted(glob.glob(os.path.join(self.root_render, '*/*/*/render')))
            train_motions = []
            for path in paths:
                splits = path.split('/')
                subject, action, frame = [splits[-4], splits[-3], int(splits[-2])]
                if subject in val_subjects:
                    val_motions.append([subject, action, frame])
                else:
                    train_motions.append([subject, action, frame])

        if self.split == 'train':
            return train_motions
        else:
            return val_motions

    def santity_check(self):
        print (f'santity check of the dataset ... before: {len(self.motion_list)} motions.')
        motion_list_valid = []
        for motion in tqdm.tqdm(self.motion_list):
            rotation = self.rotations[-1]
            subject, action, frame = motion
            if not os.path.exists(self.get_texture_path(motion, rotation)):
                continue
            if not os.path.exists(self.get_image_path(motion, rotation)):
                continue
            if not os.path.exists(self.get_mesh_path(motion)):
                continue
            if not os.path.exists(self.get_calib_path(motion, rotation)):
                continue
            if not os.path.exists(self.get_sample_path(motion)):
                continue
            if not os.path.exists(self.get_skeleton_path(motion)):
                continue
            if not os.path.exists(self.get_center_path(motion)):
                continue
            skel_path = self.get_skeleton_path(motion)
            skel = np.loadtxt(skel_path, usecols=[1, 2, 3]) / 100
            if skel[6, 1] < skel[1, 1]: # y(head) < y(hip)
                continue
            calib_path = self.get_calib_path(motion, rotation)
            calib = load_calib(calib_path)
            skel_proj = projection(skel, calib)
            if skel_proj.min() < -1.0 or skel_proj.max() > 1.0:
                continue
            motion_list_valid.append(motion)
        self.motion_list = motion_list_valid
        print (f'santity check of the dataset ... after: {len(self.motion_list)} motions.')

    def get_texture_path(self, motion, rotation):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'uv_render', f'{rotation:03d}.jpg') # be careful!

    def get_image_path(self, motion, rotation):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'render', f'{rotation:03d}.png')

    def get_mesh_path(self, motion):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'mesh_poisson.obj')

    def get_calib_path(self, motion, rotation):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'calib', f'{rotation:03d}.txt')

    def get_skeleton_path(self, motion):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'skeleton.txt')
    
    def get_center_path(self, motion):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            'center.txt')
    
    def get_sample_path(self, motion):
        subject, action, frame = motion
        return os.path.join(
            self.root_render, subject, action, f'{frame:06d}', 
            f'samples_{self.cfg.sigma_geo:.2f}_v3')
    
    def get_sampling_geo(self, motion):
        num_sample = self.cfg.num_sample_geo
        samples_path = self.get_sample_path(motion)

        samples_surface = np.load(
            os.path.join(samples_path, f'surface_{random.randint(0, 99)}.npy'))
        samples_surface = samples_surface[np.random.choice(
            samples_surface.shape[0], 4 * num_sample, replace=False)]

        samples_uniform = np.load(
            os.path.join(samples_path, f'uniform_{random.randint(0, 99)}.npy'))
        samples_uniform = samples_uniform[np.random.choice(
            samples_uniform.shape[0], num_sample // 4, replace=False)]

        samples = np.concatenate([samples_surface, samples_uniform])
        np.random.shuffle(samples)
        inside = samples[:, 3]
        samples = samples[:, 0:3]

        # balance in and out
        inside_samples = samples[inside > 0.5]
        outside_samples = samples[inside <= 0.5]

        nin = inside_samples.shape[0]
        if nin > num_sample // 2:
            inside_samples = inside_samples[:num_sample // 2]
            outside_samples = outside_samples[:num_sample // 2]
        else:
            outside_samples = outside_samples[:(num_sample - nin)]
            
        samples = np.concatenate([inside_samples, outside_samples], 0)
        labels = np.concatenate([
            np.ones(inside_samples.shape[0]), np.zeros(outside_samples.shape[0])])

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)
        return samples, labels


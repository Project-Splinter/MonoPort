import numpy as np
import torch
import torchvision
import trimesh
import os
import tqdm
import vtkplotter as vtk
from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset, load_obj_verts
from monoport.lib.dataset.utils import projection

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    dataset = PPLDynamicDataset(
        cfg.dataset, 
        mean=cfg.netG.mean, 
        std=cfg.netG.std,
        training=True)

    motion_list = dataset.motion_list
    for motion in tqdm.tqdm(motion_list):
        mesh_path = dataset.get_mesh_path(motion)
        verts = load_obj_verts(mesh_path)
        center = np.median(verts, axis=0)
        np.savetxt(
            os.path.join(os.path.dirname(mesh_path), 'center.txt'), 
            center)

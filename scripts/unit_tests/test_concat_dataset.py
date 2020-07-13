import numpy as np
import torch
import random
import torchvision
import trimesh
import vtkplotter as vtk
from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset, load_obj_verts
from monoport.lib.dataset.ppl_dynamic import load_calib as load_calib_dy
from monoport.lib.dataset.ppl_static import PPLStaticDataset
from monoport.lib.dataset.ppl_static import load_calib as load_calib_st
from monoport.lib.dataset.utils import projection

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.dataset.score_filter = 0.020
    dataset_dy = PPLDynamicDataset(
        cfg.dataset, 
        mean=cfg.netG.mean, 
        std=cfg.netG.std,
        training=False)
    dataset_st = PPLStaticDataset(
        cfg.dataset, 
        mean=cfg.netG.mean, 
        std=cfg.netG.std,
        training=True)

    while True:
        motion = random.choice(dataset_dy.motion_list)
        mesh_path = dataset_dy.get_mesh_path(motion)
        mesh = trimesh.load(mesh_path)
        
        calib_path = dataset_dy.get_calib_path(motion, rotation=90)
        calib = load_calib_dy(calib_path)

        center = np.loadtxt(dataset_dy.get_center_path(motion)).reshape(1, 3)
        center_proj = projection(center, calib).reshape(3,)
        calib[2, 3] -= center_proj[2]
        
        verts = projection(mesh.vertices, calib)
        print (np.median(verts, axis=0))
        mesh_dy = trimesh.Trimesh(verts, mesh.faces)
        mesh_dy.visual.face_colors = [200, 200, 250, 255]
        
        motion = random.choice(dataset_st.motion_list)
        mesh_path = dataset_st.get_mesh_path(motion)
        mesh = trimesh.load(mesh_path)
        calib_path = dataset_st.get_calib_path(motion, rotation=0)
        calib = load_calib_st(calib_path)
        mesh = trimesh.load(mesh_path)
        verts = projection(mesh.vertices, calib)
        print (np.median(verts, axis=0))
        mesh_st = trimesh.Trimesh(verts, mesh.faces)
        mesh_st.visual.face_colors = [250, 200, 200, 255]
        
        vtk.show(mesh_st, mesh_dy, interactive=True)
        vtk.clear()

        # break
import numpy as np
import torch
import torchvision
import trimesh
import vtkplotter as vtk
from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset
from monoport.lib.dataset.utils import projection

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    cfg.dataset.score_filter = 0.020
    cfg.dataset.scale_uniform = True
    dataset = PPLDynamicDataset(
        cfg.dataset, 
        mean=cfg.netG.mean, 
        std=cfg.netG.std,
        training=False)

    data = dataset[0]
    for k, v in data.items():
        if isinstance(v, str) or isinstance(v, int):
            print (k, v)
        else:
            print (k, v.shape)

    # load
    num_total = len(dataset)
    nrow = 4
    ncol = 4

    idxs = np.random.choice(range(num_total), size=nrow*ncol)
    images = []
    for idx in idxs:
        data = dataset[idx]
        image = data['image']
        points = projection(
            data['samples_geo'].numpy(), 
            data['calib'].numpy())
        points = (points * 0.5 + 0.5) * 512
        labels = data['labels_geo'].numpy()
        for p, l in zip(points, labels):
            x, y, z = p
            if l > 0.5:
                image[:, int(y), int(x)] = 1
        images.append(image)

        # sampled points
        vtk_list = []
        samples = data['samples_geo'].numpy()
        calib = data['calib'].numpy()
        labels = data['labels_geo'].numpy()
        samples = projection(samples, calib)
        colors = np.stack([labels, labels>0.5, labels<0.5], axis=1)
        vtk_samples = vtk.Points(samples, r=12, c=np.float32(colors))
        vtk_list.append(vtk_samples)

        # mesh
        mesh = trimesh.load(data['mesh_path'])
        verts = projection(mesh.vertices, calib)
        vtk_mesh = trimesh.Trimesh(verts, mesh.faces)
        vtk_mesh.visual.face_colors = [200, 200, 250, 255]
        vtk_list.append(vtk_mesh)
        
        vtk.show(*vtk_list, interactive=True)
        vtk.clear()
        
    images = torch.stack(images)

    # save
    torchvision.utils.save_image(
        images, './data/test_ppl_dynamic.jpg', 
        nrow=nrow, normalize=True, padding=10)

    
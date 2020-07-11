import logging
import torch
import torch.nn.functional as F
import trimesh
import tqdm
import vtkplotter as vtk

import mcubes_module as mc
from implicit_seg.functional import Seg3dTopk, Seg3dLossless
from implicit_seg.functional.utils import plot_mask3D
from monoport.lib.modeling.geometry import orthogonal

try:
    import kaolin as kal
    use_kaolin = True
except:
    use_kaolin = False

logging.getLogger('trimesh').setLevel(logging.ERROR)


def marching_cubes(vol, thresh):
    """
    vol: 3D torch tensor
    thresh: threshold
    """

    if vol.is_cuda:
        return mc.mcubes_cuda(vol, thresh)
    else:
        return mc.mcubes_cpu(vol, thresh)


def calc_metrics(mesh_pred, mesh_gt, device, sampled_points=1000):
    mesh_gt = mesh_gt.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_gt]
    mesh_gt = mesh_gt[comp_num.index(max(comp_num))]

    mesh_pred = mesh_pred.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_pred]
    mesh_pred = mesh_pred[comp_num.index(max(comp_num))]

    gt_surface_pts, _ = trimesh.sample.sample_surface_even(
        mesh_gt, sampled_points)
    pred_surface_pts, _ = trimesh.sample.sample_surface_even(
        mesh_pred, sampled_points)

    if use_kaolin:
        kal_mesh_gt = kal.rep.TriangleMesh.from_tensors(
                torch.tensor(mesh_gt.vertices).float().to(device),
                torch.tensor(mesh_gt.faces).long().to(device))
        kal_mesh_pred = kal.rep.TriangleMesh.from_tensors(
            torch.tensor(mesh_pred.vertices).float().to(device),
            torch.tensor(mesh_pred.faces).long().to(device))

        kal_distance_0 = kal.metrics.mesh.point_to_surface(
            torch.tensor(pred_surface_pts).float().to(device), kal_mesh_gt)
        kal_distance_1 = kal.metrics.mesh.point_to_surface(
            torch.tensor(gt_surface_pts).float().to(device), kal_mesh_pred)

        dist_gt_pred = torch.sqrt(kal_distance_0).cpu().numpy()
        dist_pred_gt = torch.sqrt(kal_distance_1).cpu().numpy()
    else:
        try:
            _, dist_pred_gt, _ = trimesh.proximity.closest_point(
                mesh_pred, gt_surface_pts)
            _, dist_gt_pred, _ = trimesh.proximity.closest_point(
                mesh_gt, pred_surface_pts)
        except Exception as e:
            print (e)
            return np.nan, np.nan
    
    chamfer = 0.5 * (dist_pred_gt.mean() + dist_gt_pred.mean())
    p2s = dist_pred_gt.mean()
    return chamfer, p2s


class Evaluator():
    def __init__(self, device='cuda:0'):
        ########################################
        ## variables for hierachy occupancy reconstruction
        ########################################
        self.device = device
        self.calib_tensor = torch.eye(4).unsqueeze(0).to(device)
        self.b_min = torch.tensor([-1.0, -1.0, -1.0]).float()
        self.b_max = torch.tensor([ 1.0,  1.0,  1.0]).float()
        self.resolutions = [16+1, 32+1, 64+1, 128+1, 256+1]
        self.reconEngine = Seg3dLossless(
            query_func=self.query_func, 
            b_min=self.b_min.unsqueeze(0).numpy(),
            b_max=self.b_max.unsqueeze(0).numpy(),
            resolutions=self.resolutions,
            balance_value=0.5,
            use_cuda_impl=False,
            faster=True).to(device)

    @torch.no_grad()
    def query_func(self, points, im_feat_list):
        '''
            - points: size of (bz, N, 3)
            - proj_matrix: size of (bz, 4, 4)
        return: size of (bz, 1, N)
        '''
        assert len(points) == 1
        samples = points.repeat(1, 1, 1)
        samples = samples.permute(0, 2, 1) # [bz, 3, N]

        preds = netG.query(
            im_feat_list,
            points=samples, 
            calibs=self.calib_tensor)[0]
        return preds

    @torch.no_grad()
    def recon_sdf(self, netG, image_tensor):
        netG.eval()
        im_feat_list = netG.filter(image_tensor)
        sdf = self.reconEngine(im_feat_list=im_feat_list)
        return sdf

    @torch.no_grad()
    def extract_mesh(self, sdf, calib_tensor):
        verts, faces = marching_cubes(
            F.interpolate(sdf, 256)[0, 0], 0.5) # faces normal is inversed!
        verts = (verts / 256 - 0.5) / 0.5 # o [-1, 1] 
        verts = orthogonal(
            verts.transpose(1, 0).unsqueeze(0), 
            torch.inverse(calib_tensor))[0].transpose(1, 0)
        return verts, faces

    @torch.no_grad()
    def run(self, netG, image_tensor, calib_tensor, mesh_gt, vis=False):
        netG.eval()
        sdf = self.recon_sdf(netG, image_tensor)
        verts, faces = self.extract_mesh(sdf, calib_tensor)

        mesh_pred = trimesh.Trimesh(
            verts.cpu().numpy(), faces.cpu().numpy())

        # vtkplotter visualize
        if vis:
            mesh_pred.visual.face_colors = [200, 200, 250, 255]
            vtk.show(mesh_pred, mesh_gt, interactive=True)
            vtk.clear()

        chamfer, p2s = calc_metrics(mesh_pred, mesh_gt, self.device)
        return chamfer, p2s

    @torch.no_grad()
    def run_dataloader(self, netG, data_loader, vis=False):
        chamfer_list = []
        p2s_list = []
        pbar = tqdm.tqdm(data_loader)
        for data in pbar:
            image_tensor = data['image'].to(self.device).float()
            calib_tensor = data['calib'].to(self.device).float()
            mesh_gt = trimesh.load(data['mesh_path'][0])
            chamfer, p2s = evaluator.run(
                netG, image_tensor, calib_tensor, mesh_gt, vis=vis)
            chamfer_list.append(chamfer)
            p2s_list.append(p2s)
            chamfer_avg = sum(chamfer_list) / len(chamfer_list)
            p2s_avg = sum(p2s_list) / len(p2s_list)
            pbar.set_description(
                f'chamfer: {chamfer:.3f}({chamfer_avg:.3f}) '+
                f'p2s: {p2s:.3f}({p2s_avg:.3f})')

        return chamfer_avg, p2s_avg


if __name__ == '__main__':
    import argparse
    import sys
    import os
    from monoport.lib.common.config import get_cfg_defaults
    from monoport.lib.modeling.MonoPortNet import MonoPortNet
    from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset
    from monoport.lib.dataset.ppl_static import PPLStaticDataset

    ########################################
    ## load configs
    ########################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-cfg', '--config_file', default=None, type=str, 
        help='path of the yaml config file')
    parser.add_argument(
        '-d', '--dataset', default='static', type=str, 
        choices=['static', 'dynamic', 'buff'],
        help='on which dataset to do evaluation')
    parser.add_argument(
        '-s', '--split', default='val', type=str, 
        choices=['train', 'val'],
        help='on which dataset split to do evaluation')
    parser.add_argument(
        '-v', '--vis', action='store_true',
        help='do vtk visualization')
        
    argv = sys.argv[1:sys.argv.index('--')]
    args = parser.parse_args(argv)
    opts = sys.argv[sys.argv.index('--') + 1:]

    cfg = get_cfg_defaults()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    device = 'cuda:0'

    ########################################
    ## load netG
    ########################################
    print (f'loading networkG from {cfg.netG.ckpt_path} ...')
    netG = MonoPortNet(cfg.netG)
    assert os.path.exists(cfg.netG.ckpt_path), 'we need a ckpt to run eval.'
    if 'checkpoints' in cfg.netG.ckpt_path:
        ckpt = torch.load(cfg.netG.ckpt_path, map_location="cpu")
        netG.load_state_dict(ckpt['net'])
    else:
        netG.load_legacy_pifu(cfg.netG.ckpt_path)

    netG = netG.to(device)
    netG.eval()

    ########################################
    ## build evaluator
    ########################################
    evaluator = Evaluator(device=device)

    ########################################
    ## build dataset
    ########################################
    if args.dataset == 'static':
        dataset = PPLStaticDataset(
            cfg.dataset, 
            mean=cfg.netG.mean, 
            std=cfg.netG.std,
            training=False,
            split=args.split)
    elif args.dataset == 'dynamic':
        dataset = PPLDynamicDataset(
            cfg.dataset, 
            mean=cfg.netG.mean, 
            std=cfg.netG.std,
            training=False,
            split=args.split)
    else:
        raise NotImplementedError
    
    dataset.motion_list = dataset.motion_list[:10]
    dataset.rotations = dataset.rotations[::30]
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print(
        f'data size: {len(dataset)}; '+
        f'loader size: {len(data_loader)};')

    ########################################
    ## start
    ########################################
    evaluator.run_dataloader(netG, data_loader, vis=args.vis)
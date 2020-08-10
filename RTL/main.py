import sys
import os
import argparse
import glob
import tqdm
import cv2
import math
import numpy as np
from base64 import b64encode
from sys import getsizeof

from flask import Response
from flask import Flask
from flask import render_template

import torch
import torch.nn.functional as F

from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.modeling.MonoPortNet import MonoPortNet
from monoport.lib.modeling.MonoPortNet import PIFuNetG, PIFuNetC
from monoport.lib.modeling.geometry import orthogonal, perspective
from monoport.lib.render.gl.glcontext import create_opengl_context
from monoport.lib.render.gl.AlbedoRender import AlbedoRender

import streamer_pytorch as streamer
import human_inst_seg
from implicit_seg.functional import Seg3dTopk, Seg3dLossless
from implicit_seg.functional.utils import plot_mask3D

from dataloader import DataLoader
from scene import MonoPortScene, make_rotate
from recon import pifu_calib, forward_vertices


########################################
## Global Control
########################################
DESKTOP_MODE = 'TEXTURE_NORM'
# assert DESKTOP_MODE in ['SEGM', 'NORM', 'TEXTURE', 'TEXTURE_NORM']

SERVER_MODE = 'TEXTURE'
# assert SERVER_MODE in ['NORM', 'TEXTURE']

VIEW_MODE = 'AUTO'
# assert VIEW_MODE in ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'AUTO', 'LOAD']

########################################
## load configs
########################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-cfg', '--config_file', default=None, type=str, 
    help='path of the yaml config file')
parser.add_argument(
    '--camera', action="store_true")
parser.add_argument(
    '--images', default="", nargs="*")
parser.add_argument(
    '--image_folder', default=None)
parser.add_argument(
    '--videos', default="", nargs="*")
parser.add_argument(
    '--loop', action="store_true")
parser.add_argument(
    '--use_server', action="store_true")
parser.add_argument(
    '--ip', type=str, default="192.168.1.53")
parser.add_argument(
    '--port', type=str, default="5555")


argv = sys.argv[1:sys.argv.index('--')]
args = parser.parse_args(argv)
opts = sys.argv[sys.argv.index('--') + 1:]

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(opts)
cfg.freeze()


########################################
## access avaiable GPUs
########################################
device_count = torch.cuda.device_count()
if device_count == 1:
    cuda_backbone_G='cuda:0' 
    cuda_backbone_C='cuda:0'
    cuda_recon='cuda:0'
    cuda_color='cuda:0'
elif device_count == 2:
    cuda_backbone_G='cuda:1' 
    cuda_backbone_C='cuda:1'
    cuda_recon='cuda:0'
    cuda_color='cuda:1'
else:
    raise NotImplementedError
    

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
    
netG.image_filter = netG.image_filter.to(cuda_backbone_G)
netG.surface_classifier = netG.surface_classifier.to(cuda_recon)
netG.eval()

if os.path.exists(cfg.netC.ckpt_path):
    print (f'loading networkC from {cfg.netC.ckpt_path} ...')
    netC = MonoPortNet(cfg.netC)
    netC.load_legacy_pifu(cfg.netC.ckpt_path)

    netC.image_filter = netC.image_filter.to(cuda_backbone_C)
    netC.surface_classifier = netC.surface_classifier.to(cuda_color)
    netC.eval()
else:
    netC = None
    print (f'we are not loading netC ...')


########################################
## initialize data streamer
########################################
print (f'initialize data streamer ...')
if args.camera:
    data_stream = streamer.CaptureStreamer(pad=False)
elif len(args.videos) > 0:
    data_stream = streamer.VideoListStreamer(
        args.videos * (10 if args.loop else 1))
elif len(args.images) > 0:
    data_stream = streamer.ImageListStreamer(
        args.images * (10000 if args.loop else 1))
elif args.image_folder is not None:
    images = sorted(glob.glob(args.image_folder+'/*.jpg'))
    images += sorted(glob.glob(args.image_folder+'/*.png'))
    data_stream = streamer.ImageListStreamer(
        images * (10 if args.loop else 1))


########################################
## human segmentation model
########################################
seg_engine = human_inst_seg.Segmentation(
    device=cuda_backbone_G, verbose=False)
seg_engine.eval()


########################################
## pre-loaded scene for rendering
########################################
scene = MonoPortScene(size=(256, 256))


########################################
## variables for hierachy occupancy reconstruction
########################################
calib_tensor = torch.eye(4).unsqueeze(0).to(cuda_recon)
@torch.no_grad()
def query_func(points, im_feat_list, calib_tensor):
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
        calibs=calib_tensor)[0]
    return preds

b_min = torch.tensor([-1.0, -1.0, -1.0]).float()
b_max = torch.tensor([ 1.0,  1.0,  1.0]).float()
resolutions = [16+1, 32+1, 64+1, 128+1, 256+1]
reconEngine = Seg3dLossless(
    query_func=query_func, 
    b_min=b_min.unsqueeze(0).numpy(),
    b_max=b_max.unsqueeze(0).numpy(),
    resolutions=resolutions,
    balance_value=0.5,
    use_cuda_impl=False,
    faster=True).to(cuda_recon)


########################################
## variables for color inference
########################################
canvas = torch.ones(
    (resolutions[-1], resolutions[-1], 3), dtype=torch.float32
).to(cuda_color) 
mat = torch.eye(4, dtype=torch.float32)
length = b_max - b_min
mat[0, 0] = length[0] / resolutions[-1]
mat[1, 1] = length[1] / resolutions[-1]
mat[2, 2] = length[2] / resolutions[-1]
mat[0:3, 3] = b_min
mat_color = mat.to(cuda_color)

@torch.no_grad()
def colorization(netC, feat_tensor_C, X, Y, Z, calib_tensor, norm=None):
    if X is None:
        return None

    device = calib_tensor.device
    global canvas
    # use normal as color
    if norm is not None:
        color = (norm + 1) / 2 
        color = color.clamp(0, 1)
        image = canvas.clone()
        image[X, Y, :] = color
        return image

    # use netC to predict color
    else:            
        feat_tensor_C = [
            [feat.to(device) for feat in feats] for feats in feat_tensor_C]
        verts = torch.stack([
            X.float(), Y.float(), resolutions[-1]-Z.float() # TODO
        ], dim=1)

        samples = verts.unsqueeze(0).repeat(1, 1, 1)
        samples = samples.permute(0, 2, 1) # [bz, 3, N]
        samples = orthogonal(samples, mat_color.unsqueeze(0))

        preds = netC.query(
            feat_tensor_C,
            points=samples, 
            calibs=calib_tensor)[0]
        
        color = preds[0] * 0.5 + 0.5 # FIXME
        color = color.t() # [N, 3]
    
        image = canvas.clone()
        image[X, Y, :] = color
        return image


@torch.no_grad()
def visulization(render_norm, render_tex=None):
    if render_norm is None and render_tex is None:
        return None, None, None

    render_size = 256

    if render_norm is not None:
        render_norm = render_norm.detach() * 255.0
        render_norm = torch.rot90(render_norm, 1, [0, 1]).permute(2, 0, 1).unsqueeze(0)
        render_norm = F.interpolate(render_norm, size=(render_size, render_size))
        render_norm = render_norm[0].cpu().numpy().transpose(1, 2, 0)
        reference = render_norm

    if render_tex is not None:
        render_tex = render_tex.detach() * 255.0
        render_tex = torch.rot90(render_tex, 1, [0, 1]).permute(2, 0, 1).unsqueeze(0)
        render_tex = F.interpolate(render_tex, size=(render_size, render_size))
        render_tex = render_tex[0].cpu().numpy().transpose(1, 2, 0)
        reference = render_tex

    bg = np.logical_and(
        np.logical_and(
            reference[:, :, 0] == 255,
            reference[:, :, 1] == 255),
        reference[:, :, 2] == 255,
    ).reshape(render_size, render_size, 1)
    mask = ~bg

    return render_norm, render_tex, mask



########################################
## define async processors
########################################
mean = torch.tensor(cfg.netG.mean).to(cuda_backbone_G).view(1, 3, 1, 1)
std = torch.tensor(cfg.netG.std).to(cuda_backbone_G).view(1, 3, 1, 1)
scaled_boxes = [torch.Tensor([[ 50.0,  0.0, 450.0, 500.0]]).to(cuda_backbone_G)]

def update_camera():
    extrinsic = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
    if VIEW_MODE == 'FRONT':
        yaw, pitch = 20, 0  
    elif VIEW_MODE == 'BACK':
        yaw, pitch = 20, 180  
    elif VIEW_MODE == 'LEFT':
        yaw, pitch = 20, 90   
    elif VIEW_MODE == 'RIGHT':
        yaw, pitch = 20, 270
    elif VIEW_MODE == 'AUTO':
        extrinsic, intrinsic = scene.update_camera(load=False)
        return extrinsic, intrinsic
    elif VIEW_MODE == 'LOAD':
        extrinsic, intrinsic = scene.update_camera(load=True)
        return extrinsic, intrinsic
    else:
        raise NotImplementedError

    intrinsic = scene.intrinsic
    R = np.matmul(
        make_rotate(math.radians(yaw), 0, 0), 
        make_rotate(0, math.radians(pitch), 0)
        )
    extrinsic[0:3, 0:3] = R 
    return extrinsic, intrinsic
     

processors=[
    lambda data: {"input": data.to(cuda_backbone_G, non_blocking=True)},

    # scene camera updating
    lambda data_dict: {
        **data_dict, 
        **dict(zip(
            ["extrinsic", "intrinsic"], 
            update_camera(),
        ))},

    # calculate calib tensor
    lambda data_dict: {
        **data_dict, 
        "calib_tensor": pifu_calib(
            data_dict["extrinsic"], data_dict["intrinsic"], device=cuda_recon)
        },  
    
    # instance segmentation:
    lambda data_dict: {
        **data_dict, 
        **dict(zip(
            ["segm", "bboxes", "probs"], 
            seg_engine(data_dict["input"], scaled_boxes)
        ))},

    # update input by removing bg
    lambda data_dict: {
        **data_dict, 
        "input_netG": (
            ((data_dict["segm"][:, 0:3] * 0.5 + 0.5) - mean) / std
            )*data_dict["segm"][:, 3:4]
        }, 

    # update input by removing bg
    lambda data_dict: {
        **data_dict, 
        "input_netC": data_dict["segm"][:, 0:3] * data_dict["segm"][:, 3:4]
        },  

    # pifu netG feature extraction
    lambda data_dict: {
        **data_dict, 
        "feat_tensor_G": netG.filter(data_dict["input_netG"])
        }, 

    # pifu netC feature extraction
    lambda data_dict: {
        **data_dict, 
        "feat_tensor_C": netC.filter(
            data_dict["input_netC"].to(cuda_backbone_C, non_blocking=True),
            feat_prior=data_dict["feat_tensor_G"][-1][-1]) \
                if (netC is not None) and ('TEXTURE' in DESKTOP_MODE or 'TEXTURE' in SERVER_MODE)  else None
        }, 

    # move feature to cuda_recon device
    lambda data_dict: {
        **data_dict, 
        "feat_tensor_G": [
            [feat.to(cuda_recon) for feat in feats]
            for feats in data_dict["feat_tensor_G"]]
        }, 

    # pifu sdf space recon
    lambda data_dict: {
        **data_dict, 
        "sdf": reconEngine(
            im_feat_list=data_dict["feat_tensor_G"],
            calib_tensor=data_dict["calib_tensor"])
        },  

    # lambda data_dict: plot_mask3D(
    #     data_dict["sdf"].to("cpu"), title="sdf"),
    
    # pifu visible vertices
    lambda data_dict: {
        **data_dict, 
        **dict(zip(
            ["X", "Y", "Z", "norm"], 
            forward_vertices(data_dict["sdf"], direction="front")
        ))},  

    lambda data_dict: {
        **data_dict, 
        "X": data_dict['X'].to(cuda_color) if data_dict['X'] is not None else None,
        "Y": data_dict['Y'].to(cuda_color) if data_dict['X'] is not None else None,
        "Z": data_dict['Z'].to(cuda_color) if data_dict['X'] is not None else None,
        "norm": data_dict['norm'].to(cuda_color) if data_dict['X'] is not None else None,
        "calib_tensor": data_dict['calib_tensor'].to(cuda_color) if data_dict['X'] is not None else None,
        },  

    # pifu render normal
    lambda data_dict: {
        **data_dict, 
        "render_norm": colorization(
            netC,
            None,
            data_dict["X"],
            data_dict["Y"],
            data_dict["Z"],
            data_dict["calib_tensor"],
            data_dict["norm"]) if ('NORM' in DESKTOP_MODE or 'NORM' in SERVER_MODE)  else None
        },  

    # pifu render texture
    lambda data_dict: {
        **data_dict, 
        "render_tex": colorization(
            netC,
            data_dict["feat_tensor_C"],
            data_dict["X"],
            data_dict["Y"],
            data_dict["Z"],
            data_dict["calib_tensor"],
            None) if data_dict["feat_tensor_C"] else None
        },

    # visualization
    lambda data_dict: {
        **data_dict, 
        **dict(zip(
            ["render_norm", "render_tex", "mask"], 
            visulization(
                data_dict["render_norm"],
                data_dict["render_tex"])
        ))},
]


########################################
## build async processor
########################################
loader = DataLoader(
    data_stream, 
    batch_size=1, 
    num_workers=1, 
    pin_memory=True,
    processors=processors,
)


def main_loop():
    global DESKTOP_MODE, SERVER_MODE, VIEW_MODE

    window_server = np.ones((256, 256, 3), dtype=np.uint8) * 255
    window_desktop = np.ones((512, 1024, 3), dtype=np.uint8) * 255

    create_opengl_context(256, 256)
    renderer = AlbedoRender(width=256, height=256, multi_sample_rate=1)
    renderer.set_attrib(0, scene.vert_data)
    renderer.set_attrib(1, scene.uv_data)
    renderer.set_texture('TargetTexture', scene.texture_image)

    def render(extrinsic, intrinsic):
        uniform_dict = {'ModelMat': extrinsic, 'PerspMat': intrinsic}
        renderer.draw(uniform_dict)
        color = (renderer.get_color() * 255).astype(np.uint8)
        background = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        background = cv2.resize(background, (256, 256))
        return background

    for data_dict in tqdm.tqdm(loader):
        render_norm = data_dict["render_norm"] # [256, 256, 3] RGB
        render_tex = data_dict["render_tex"] # [256, 256, 3] RGB
        mask = data_dict["mask"]
        extrinsic = data_dict["extrinsic"]
        intrinsic = data_dict["intrinsic"]
        
        if DESKTOP_MODE is not None:
            input4c = data_dict["segm"].cpu().numpy()[0].transpose(1, 2, 0) # [512, 512, 4]
            input = (input4c[:, :, 0:3] * 0.5) + 0.5
        if DESKTOP_MODE == 'SEGM':
            segmentation = (input4c[:, :, 0:3] * input4c[:, :, 3:4] * 0.5) + 0.5
            window_desktop = np.uint8(np.hstack([
                input * 255, 
                segmentation * 255
                ])) # RGB
        elif DESKTOP_MODE == 'NORM':
            if render_norm is None:
                render_norm = np.ones((256, 256, 3), dtype=np.float32) * 255
            window_desktop = np.uint8(np.hstack([
                input * 255, 
                cv2.resize(render_norm, (512, 512))
                ])) # RGB
        elif DESKTOP_MODE == 'TEXTURE':
            if render_tex is None:
                render_tex = np.ones((256, 256, 3), dtype=np.float32) * 255
            window_desktop = np.uint8(np.hstack([
                input * 255, 
                cv2.resize(render_tex, (512, 512))
                ])) # RGB
        elif DESKTOP_MODE == 'TEXTURE_NORM':
            if render_tex is None:
                render_tex = np.ones((256, 256, 3), dtype=np.float32) * 255
            if render_norm is None:
                render_norm = np.ones((256, 256, 3), dtype=np.float32) * 255
            window_desktop = np.uint8(np.vstack([
                render_tex,
                render_norm
                ])) # RGB
        else:
            window_desktop = None

        if DESKTOP_MODE is not None:
            # window_desktop = cv2.resize(window_desktop, (2400, 1200))
            cv2.namedWindow('window_desktop', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('window_desktop', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window_desktop', window_desktop[:, :, ::-1])
        
        if args.use_server:
            if SERVER_MODE == 'NORM':
                background = render(extrinsic, intrinsic)
                if mask is None:
                    window_server = background
                else:
                    window_server = np.uint8(mask * render_norm + (1 - mask) * background)
            elif SERVER_MODE == 'TEXTURE':
                background = render(extrinsic, intrinsic)
                if mask is None:
                    window_server = background
                else:
                    window_server = np.uint8(mask * render_tex + (1 - mask) * background)  
            else:
                if render_norm is not None:
                    window_server = np.uint8(render_norm)      
            
            # yield window_desktop, window_server
            (flag, encodedImage) = cv2.imencode(".jpg", window_server[:, :, ::-1])
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')

        key = cv2.waitKey(1)
        if key == ord('q'):
            DESKTOP_MODE = 'SEGM'
        elif key == ord('w'):
            DESKTOP_MODE = 'NORM'
        elif key == ord('e'):
            DESKTOP_MODE = 'TEXTURE'
        elif key == ord('r'):
            DESKTOP_MODE = 'TEXTURE_NORM'

        # elif key == ord('a'):
        #     SERVER_MODE = 'SEGM'
        elif key == ord('s'):
            SERVER_MODE = 'NORM'
        elif key == ord('d'):
            SERVER_MODE = 'TEXTURE'
        elif key == ord('f'):
            SERVER_MODE = None

        elif key == ord('z'):
            VIEW_MODE = 'FRONT'
        elif key == ord('x'):
            VIEW_MODE = 'BACK'
        elif key == ord('c'):
            VIEW_MODE = 'LEFT'
        elif key == ord('v'):
            VIEW_MODE = 'RIGHT'
        elif key == ord('b'):
            VIEW_MODE = 'AUTO'
        elif key == ord('n'):
            VIEW_MODE = 'LOAD'
        
        

if __name__ == '__main__':
    if args.use_server:
        ########################################
        ## Flask related
        ########################################
        app = Flask(__name__)

        @app.route("/")
        def index():
            return render_template("test_flask.html")
        
        @app.route("/video_feed")
        def video_feed():
            return Response(main_loop(),
                mimetype = "multipart/x-mixed-replace; boundary=frame")

        # start the flask app
        app.run(host=args.ip, port=args.port, debug=True,
            threaded=True, use_reloader=False)
    else:
        print('start main_loop.')
        for _ in main_loop():
            pass

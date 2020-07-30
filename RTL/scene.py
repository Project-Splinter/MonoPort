import cv2
import numpy as np
import math
import os
import json

from monoport.lib.render.gl.glcontext import create_opengl_context
from monoport.lib.render.gl.AlbedoRender import AlbedoRender
from monoport.lib.render.BaseCamera import BaseCamera
from monoport.lib.render.PespectiveCamera import PersPectiveCamera
from monoport.lib.render.CameraPose import CameraPose

from monoport.lib.mesh_util import load_obj_mesh


_RTL_DATA_FOLDER = os.path.join(
    os.path.dirname(__file__), '../data/RTL/')


def _load_grass(grass_size=3.0, grass_center=np.array([0, -0.9, 0])):
    mesh_file = os.path.join(
        _RTL_DATA_FOLDER, 
        'grass/10438_Circular_Grass_Patch_v1_iterations-2.obj')
    text_file = os.path.join(
        _RTL_DATA_FOLDER,
        'grass/10438_Circular_Grass_Patch_v1_Diffuse.jpg')
    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(
        mesh_file, with_normal=True, with_texture=True)
    vertices = vertices[:, [0, 2, 1]]

    # change cm to meter
    vertices = vertices / 150 * grass_size
    vertices = vertices - vertices.mean(axis=0)
    vertices += grass_center

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    # Here we pack the vertex data needed for the render
    vert_data = vertices[faces.reshape([-1])]
    uv_data = textures[face_textures.reshape([-1])]
    return vert_data, uv_data, texture_image
    

def _load_intrinsic(near=0.0, far=10.0, scale=2.0):
    intrinsic_cam = BaseCamera()
    intrinsic_cam.near = near
    intrinsic_cam.far = far
    intrinsic_cam.set_parameters(scale, scale)
    return intrinsic_cam.get_projection_mat()


def _load_extrinsic():
    path = os.path.join(
        _RTL_DATA_FOLDER, 'webxr/modelview.json')
    with open(path, 'r') as f:
        extrinsic = json.load(f)['data']
    extrinsic = np.array(extrinsic).reshape(4, 4).T
    return extrinsic


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class MonoPortScene:
    def __init__(self, size=(512, 512)):
        self.vert_data, self.uv_data, self.texture_image = _load_grass()
        self.intrinsic = _load_intrinsic()

        # create_opengl_context(size[0], size[1])
        # self.renderer = AlbedoRender(width=size[0], height=size[1], multi_sample_rate=1)
        # self.renderer.set_attrib(0, self.vert_data)
        # self.renderer.set_attrib(1, self.uv_data)
        # self.renderer.set_texture('TargetTexture', self.texture_image)

        self.extrinsic = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -2.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.step = 0

    def update_camera(self, load=False):
        if load == False:
            if self.step < 3600000:
                yaw = 20
                pitch = self.step
            else:
                yaw = self.step % 180
                pitch = 0

            R = np.matmul(
                make_rotate(math.radians(yaw), 0, 0), 
                make_rotate(0, math.radians(pitch), 0)
                )
            self.extrinsic[0:3, 0:3] = R 
            self.step += 3
            extrinsic = self.extrinsic
        else:
            while True:
                try:
                    extrinsic = _load_extrinsic()
                    break
                except Exception as e:
                    print (e)
        
        return extrinsic, self.intrinsic

    def render(self, extrinsic, intrinsic):
        uniform_dict = {
            'ModelMat': extrinsic,
            'PerspMat': intrinsic,
        }
        self.renderer.draw(
            uniform_dict
        )

        color = (self.renderer.get_color() * 255).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return color


if __name__ == '__main__':
    import tqdm
    scene = MonoPortScene()
    
    for _ in tqdm.tqdm(range(10000)):
        extrinsic, intrinsic = scene.update_camera()
        background = scene.render(extrinsic, intrinsic)
        # print (extrinsic)
        cv2.imshow('scene', background)
        cv2.waitKey(15)
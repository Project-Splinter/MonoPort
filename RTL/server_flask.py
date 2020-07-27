from flask import Response
from flask import Flask
from flask import render_template

import threading
import argparse
import datetime
import time
import cv2
import glob
import tqdm
import numpy as np
import os
from base64 import b64encode
from sys import getsizeof
from main import loader, scene

from monoport.lib.render.gl.glcontext import create_opengl_context
from monoport.lib.render.gl.AlbedoRender import AlbedoRender

# initialize a flask object
app = Flask(__name__)


########################################
## Global Control
########################################
DESKTOP_MODE = 'NORM'
# assert DESKTOP_MODE in ['SEGM', 'NORM', 'TEXURE']

SERVER_MODE = 'NONE'
# assert SERVER_MODE in ['NORM', 'TEXTURE']


def img_base64(img_path):
    with open(img_path,"rb") as f:
        data = f.read()
        print("data:", getsizeof(data))
        assert data[-2:] == b'\xff\xd9'
        base64_str = b64encode(data).decode('utf-8')
        print("base64:", getsizeof(base64_str))
    return base64_str


@app.route("/")
def index():
    # return the rendered template
    return render_template("test_flask.html")


def generate():
    global DESKTOP_MODE, SERVER_MODE

    window_server = np.ones((256, 256, 3), dtype=np.uint8) * 255
    window_desktop = np.ones((512, 1024, 3), dtype=np.uint8) * 255

    create_opengl_context(256, 256)
    renderer = AlbedoRender(width=256, height=256, multi_sample_rate=1)
    renderer.set_attrib(0, scene.vert_data)
    renderer.set_attrib(1, scene.uv_data)
    renderer.set_texture('TargetTexture', scene.texture_image)

    def render(extrinsic, intrinsic):
        renderer.set_texture('TargetTexture', scene.texture_image)
        uniform_dict = {'ModelMat': extrinsic, 'PerspMat': intrinsic}
        renderer.draw(uniform_dict)
        color = (renderer.get_color() * 255).astype(np.uint8)
        background = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
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
                render_norm = np.zeros((512, 512, 3), dtype=np.float32)
            window_desktop = np.uint8(np.hstack([
                input * 255, 
                cv2.resize(render_norm, (512, 512))
                ])) # RGB
        elif DESKTOP_MODE == 'TEXTURE':
            if render_tex is None:
                render_tex = np.zeros((512, 512, 3), dtype=np.float32)
            window_desktop = np.uint8(np.hstack([
                input * 255, 
                cv2.resize(render_tex, (512, 512))
                ])) # RGB
        else:
            window_desktop = None

        if DESKTOP_MODE is not None:
            window_desktop = cv2.resize(window_desktop, (2400, 1200))
            cv2.imshow('window_desktop', window_desktop[:, :, ::-1])
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            DESKTOP_MODE = 'SEGM'
        elif key == ord('w'):
            DESKTOP_MODE = 'NORM'
        elif key == ord('e'):
            DESKTOP_MODE = 'TEXTURE'
        elif key == ord('r'):
            DESKTOP_MODE = None

        elif key == ord('a'):
            SERVER_MODE = 'SEGM'
        elif key == ord('s'):
            SERVER_MODE = 'NORM'
        elif key == ord('d'):
            SERVER_MODE = 'TEXTURE'
        elif key == ord('f'):
            SERVER_MODE = None
        
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


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start the flask app
    app.run(host="192.168.1.232", port="5555", debug=True,
        threaded=True, use_reloader=False)

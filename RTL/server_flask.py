from flask import Response
from flask import Flask
from flask import render_template

import threading
import argparse
import datetime
import time
import cv2
import glob
import os
from base64 import b64encode
from sys import getsizeof
from main import *


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


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
    # global _IMAGES_IDX
    # while True:
    #     filename = _IMAGES[_IMAGES_IDX % len(_IMAGES)]
    #     _IMAGES_IDX += 1
    #     image = cv2.imread(filename)
    for data_dict in loader:
        extrinsic = data_dict["extrinsic"]
        intrinsic = data_dict["intrinsic"]
        
        # background = scene.render(extrinsic, intrinsic)
        background = np.zeros((256, 256, 3), dtype=np.float32)

        render_norm = data_dict["render_norm"]
        render_tex = data_dict["render_tex"]
        mask = data_dict["mask"]

        render_norm = np.uint8(mask * render_norm + (1 - mask) * background)
        if render_tex is not None:
            render_tex = np.uint8(mask * render_tex + (1 - mask) * background)
            window = np.hstack([render_norm, render_tex])
        else:
            window = render_norm

        # yield window
        (flag, encodedImage) = cv2.imencode(".jpg", window)
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
    # # start a thread that will perform motion detection
    # t = threading.Thread(target=detect_motion, args=(
    #     args["frame_count"],))
    # t.daemon = True
    # t.start()

    # start the flask app
    app.run(host="192.168.1.232", port="5555", debug=True,
        threaded=True, use_reloader=False)

    for window in main_loop():
        cv2.imshow('window', window)
        cv2.waitKey(1)

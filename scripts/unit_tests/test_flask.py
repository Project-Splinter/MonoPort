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

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

_IMAGES = sorted(glob.glob(
    '/home/rui/local/projects/PIFu-RealTime/zenTelePort/data/recording/test/*.jpg'))
_IMAGES_IDX = 0


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
    global _IMAGES_IDX
    while True:
        filename = _IMAGES[_IMAGES_IDX % len(_IMAGES)]
        _IMAGES_IDX += 1
        image = cv2.imread(filename)
        (flag, encodedImage) = cv2.imencode(".jpg", image)
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
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="192.168.1.232",
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default="5555",
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # # start a thread that will perform motion detection
    # t = threading.Thread(target=detect_motion, args=(
    #     args["frame_count"],))
    # t.daemon = True
    # t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# # release the video stream pointer
# vs.stop()

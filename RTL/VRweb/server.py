import argparse
import os
import json
import numpy as np
from base64 import b64encode
from PIL import Image
from skimage import io
from sys import getsizeof
import cv2

import tornado.web
import tornado.websocket
import tornado.httpserver


def img_base64(img_path):
    with open(img_path,"rb") as f:
        data = f.read()
        print("data:", getsizeof(data))
        assert data[-2:] == b'\xff\xd9'
        base64_str = b64encode(data).decode('utf-8')
        print("base64:", getsizeof(base64_str))
    return base64_str

def img_bytes(img_path):
    with open(img_path,"rb") as f:
        data = f.read()
        assert data[-2:] == b'\xff\xd9'
    return data



class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("index.html", port=args.port)

_mode = 0

class WebSocket(tornado.websocket.WebSocketHandler):
    users = set()

    def open(self):
        print ('user connected!')
        WebSocket.users.add(self)

    def on_message(self, message):
        json_rpc = json.loads(message)
        if json_rpc["name"] == "switch":
            global _mode
            _mode = (_mode + 1) % 2 

    @classmethod
    def push(cls):
        for user in cls.users:
            print ('push!')
            global _mode
            if _mode == 0:
                filename = './data/texture.jpg'
            else:
                filename = './data/normal.jpg'

            try:
                img_tex = img_bytes(filename)
            except Exception as e:
                print (e)
                continue
                
            user.write_message(message=img_tex, binary=True)
            
# python server.py --port 8001 --cert ruilong
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a webserver for stuff.")
    parser.add_argument(
        "--port", type=int, default=8001, help="The port on which to serve the website.")
    parser.add_argument(
        "--cert", type=str, default="ruilong", choices=["ruilong", "yuliang"], help="Whose cert to be used?")
    args = parser.parse_args()

    handlers = [(r"/", IndexHandler), (r"/websocket", WebSocket),
            (r'/static/(.*)', tornado.web.StaticFileHandler,
             {'path': os.path.normpath(os.path.dirname(__file__))})]
    application = tornado.web.Application(handlers)

    http_server = tornado.httpserver.HTTPServer(application, ssl_options={
            "certfile": f"./cert/{args.cert}.cert",
            "keyfile": f"./cert/{args.cert}.key",
        })

    http_server.listen(args.port)
    print ('server start!')
    tornado.ioloop.PeriodicCallback(WebSocket.push, 100).start()
    tornado.ioloop.IOLoop.instance().start()

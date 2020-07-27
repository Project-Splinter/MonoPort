import argparse
import os
import json
import time
import numpy as np
from base64 import b64encode
from PIL import Image

import tornado.web
import tornado.websocket
import tornado.httpserver


_RTL_DATA_FOLDER = os.path.join(
    os.path.dirname(__file__), '../../data/RTL/')


def from_dict_to_json(data):
    arr = np.zeros((4,4))
    for key in data.keys():
        ind = int(key)
        arr[ind//4, ind%4] = data[key]
    return {"data":arr.tolist()}


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index_webxr.html", port=args.port)


class WebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        print ('user connected!')
        self.timer = 0

    def on_message(self, message):
        timer = time.time() # sec
        if timer - self.timer > 0.1:
            self.timer = timer
            """Evaluates the function pointed to by json-rpc."""
            json_rpc = json.loads(message)
            print (timer, json_rpc["name"])
            path = os.path.join(
                _RTL_DATA_FOLDER, f'webxr/{json_rpc["name"]}.json')
            with open(path, 'w') as f:
                json.dump(from_dict_to_json(json_rpc["data"]), f, indent=4)

            
# python server_webxr.py --port 8000 --cert ruilong
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Starts a webserver for stuff.")
    parser.add_argument(
        "--port", type=int, default=8000, help="The port on which to serve the website.")
    parser.add_argument(
        "--cert", type=str, default="ruilong", choices=["ruilong", "yuliang"], help="Whose cert to be used?")
    args = parser.parse_args()

    handlers = [(r"/", IndexHandler), (r"/websocket", WebSocket),
            (r'/static/(.*)', tornado.web.StaticFileHandler,
             {'path': os.path.normpath(os.path.dirname(__file__))})]
    application = tornado.web.Application(handlers)

    http_server = tornado.httpserver.HTTPServer(application, ssl_options={
            "certfile": os.path.join(os.path.dirname(__file__), f"./cert/{args.cert}.cert"),
            "keyfile": os.path.join(os.path.dirname(__file__), f"./cert/{args.cert}.key"),
        })

    http_server.listen(args.port)
    if args.cert == 'ruilong':
        print (f'server start at: https://www/liruilong.codes:{args.port}')
    tornado.ioloop.IOLoop.instance().start()

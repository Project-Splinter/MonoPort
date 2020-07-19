#!/usr/bin/env python3
# Usage: first create server.
#   python test_server.py
# Then and interavtive client.
#   python -m websockets ws://127.0.0.1:5555/
import asyncio
import websockets
import threading
import time
import random
import glob
from base64 import b64encode
from sys import getsizeof


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


def gen_data():
    time.sleep(1)
    # data = random.randint(1, 10)
    # print(f"Generating data... {data}")

    global _IMAGES_IDX
    filename = _IMAGES[_IMAGES_IDX]
    _IMAGES_IDX += 1
    image = img_base64(filename)
    print (getsizeof(image))
    # image = str([filename] * 100000)
    # try:
    #     image = img_base64(filename)
    # except Exception as e:
    #     print (e)
    # print (filename)
    # image = random.randint(1, 10)
    return image

async def send(client, data):
    await client.send(data)

async def handler(client, path):
    # Register.
    print("Websocket Client Connected.", client)
    clients.append(client)
    while True:
        try:
            print("ping", client)
            pong_waiter = await client.ping()
            await pong_waiter
            print("pong", client)
            time.sleep(3)
        except Exception as e:
            clients.remove(client)
            print(f"Websocket Client Disconnected due to {e}", client)
            break

clients = []
start_server = websockets.serve(
    handler, "192.168.1.232", 5555, write_limit=2**17)

asyncio.get_event_loop().run_until_complete(start_server)
threading.Thread(target = asyncio.get_event_loop().run_forever).start()

print("Socket Server Running. Starting main loop.")

while True:
    message_clients = clients.copy()
    for client in message_clients:
        data = str(gen_data())
        print("Sending ...")
        try:
            asyncio.run(send(client, data))
        except:
            # Clients might have disconnected during the messaging process,
            # just ignore that, they will have been removed already.
            pass

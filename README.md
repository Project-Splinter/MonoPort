# Monoport

Intro.

## How to run our Siggraph RTL Demo

#### 0. Setup the repo (TODO: add dependences into setup.py)
```
python setup.py develop
```

#### 1. Start the main process as a server. 
```
# if you want to use the input from a webcam:
python RTL/main.py --use_server --ip <YOUR_IP_ADDRESS> --port 5555 --camera -- netG.ckpt_path ./data/PIFu/net_G netC.ckpt_path ./data/PIFu/net_C

# or if you want to use the input from a image folder:
python RTL/main.py --use_server --ip <YOUR_IP_ADDRESS> --port 5555 --image_folder <IMAGE_FOLDER> -- netG.ckpt_path ./data/PIFu/net_G netC.ckpt_path ./data/PIFu/net_C

# or if you want to use the input from a video:
python RTL/main.py --use_server --ip <YOUR_IP_ADDRESS> --port 5555 --videos <VIDEO_PATH> -- netG.ckpt_path ./data/PIFu/net_G netC.ckpt_path ./data/PIFu/net_C
```

If everything goes well, you should be able to see those logs after waiting for a few seconds:

    loading networkG from ./data/PIFu/net_G ...
    loading networkC from ./data/PIFu/net_C ...
    initialize data streamer ...
    Using cache found in /home/rui/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub
    Using cache found in /home/rui/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub
    * Serving Flask app "main" (lazy loading)
    * Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    * Debug mode: on
    * Running on http://<YOUR_IP_ADDRESS>:5555/ (Press CTRL+C to quit)

#### 2. Access the server to start.
Open the page `http://<YOUR_IP_ADDRESS>:5555/` on a web browser from any device (Desktop/IPad/IPhone), You should be able to see the **MonoPort VR Demo** page on that device, and at the same time you should be able to see the a screen poping up on your desktop, showing the reconstructed normal and texture image.

#### 3. Play with VR demo. (TODO: bc of the https cert, this step is not easy for the public to use)
As a VR prototype, this system also allow users to control the camera in the **MonoPort VR Demo** using the sensor from IPad/IPhone. To achieve that, you need to start another server :
```
python RTL/VRweb/server_webxr.py --port 8000 --cert ruilong
```

Then you can access `https://www/liruilong.codes:8000` from the app **XRViewer**, and click the button **Enter WebXR** in the page. From that moment, your mobile device would become a camera in our VR scene, and you can move your mobile device around to observe the reconstructed human.

## Citation
# [Monoport: Monocular Volumetric Human Teleportation (SIGGRAPH 2020 Real-Time Live)](http://xiuyuliang.cn/monoport/)

### Time: Tuesday, 25 August 2020 (Pacific Time Zone)
[![report](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2007.13988) [![homepage](https://img.shields.io/badge/Project-Homepage-green)](http://xiuyuliang.cn/monoport/) [![report](https://img.shields.io/badge/Demo-Youtube-yellow)](https://youtu.be/fQDsYVE7GtQ)

Existing volumetric capture systems require many cameras and lengthy post processing. We introduce the first system that can capture a completely clothed human body (including the back) using a single RGB webcam and in real time. Our deep-learning-based approach enables new possibilities for low-cost and consumer-accessible immersive teleportation.

<p align='center'>
    <img src='figs/rtl.jpg'/>
</p>

## Requirements
- Python 3.7
- [PyTorch](https://pytorch.org/) tested on 1.4.0
- [ImplicitSegCUDA](https://github.com/Project-Splinter/ImplicitSegCUDA)
- [human_inst_seg](https://github.com/Project-Splinter/human_inst_seg)
- [streamer_pytorch](https://github.com/Project-Splinter/streamer_pytorch)
- [human_det](https://github.com/Project-Splinter/human_det)

## How to run our Siggraph RTL Demo

#### 0. Setup the repo
First you need to download the model into './data' folder:
```
mkdir -p data/PIFu/
cd data/PIFu/
wget "https://drive.google.com/uc?export=download&id=1zEmVXG2VHy0MMzngcRshB4D8Sr_oLHsm" -O net_G
wget "https://drive.google.com/uc?export=download&id=1V83B6GDIjYMfHdpg-KcCSAPgHxpafHgd" -O net_C
cd ../../
```

Then setup this repo:
```
pip install -r requirements.txt
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

<p align='center'>
    <img src='figs/twoside.png'/>
</p>

#### 3. Play with VR demo. (TODO: bc of the https cert, this step is not easy for the public to use)
As a VR prototype, this system also allow users to control the camera in the **MonoPort VR Demo** using the sensor from IPad/IPhone. To achieve that, you need to start another server :
```
python RTL/VRweb/server_webxr.py --port 8000 --cert ruilong
```

Then you can access `https://www/liruilong.codes:8000` from the app **XRViewer**, and click the button **Enter WebXR** in the page. From that moment, your mobile device would become a camera in our VR scene, and you can move your mobile device around to observe the reconstructed human.

## Contributors

MonoPort is based on [Monocular Real-Time Volumetric Performance Capture(ECCV'20)](http://xiuyuliang.cn/monoport/), authored by Ruilong Li*([@liruilong940607](https://github.com/liruilong940607)), Yuliang Xiu*([@yuliangxiu](https://github.com/YuliangXiu)), Shunsuke Saito([@shunsukesaito](https://github.com/shunsukesaito)), Zeng Huang([@ImaginationZ](https://github.com/ImaginationZ)) and Kyle Olszewski([@kyleolsz](https://github.com/kyleolsz)), [Hao Li](https://www.hao-li.com/) is the corresponding author.


## Citation

```
@article{li2020monocular,
    title={Monocular Real-Time Volumetric Performance Capture},
    author={Li, Ruilong and Xiu, Yuliang and Saito, Shunsuke and Huang, Zeng and Olszewski, Kyle and Li, Hao},
    journal={arXiv preprint arXiv:2007.13988},
    year={2020}
  }
```

## Relevant Works

**[PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization (ICCV 2019)](https://shunsukesaito.github.io/PIFu/)**  
*Shunsuke Saito\*, Zeng Huang\*, Ryota Natsume\*, Shigeo Morishima, Angjoo Kanazawa, Hao Li*

The original work of Pixel-Aligned Implicit Function for geometry and texture reconstruction, unifying sigle-view and multi-view methods.

**[PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (CVPR 2020)](https://shunsukesaito.github.io/PIFuHD/)**  
*Shunsuke Saito, Tomas Simon, Jason Saragih, Hanbyul Joo*

They further improve the quality of reconstruction by leveraging multi-level approach!

**[ARCH: Animatable Reconstruction of Clothed Humans (CVPR 2020)](https://arxiv.org/pdf/2004.04572.pdf)**  
*Zeng Huang, Yuanlu Xu, Christoph Lassner, Hao Li, Tony Tung*

Learning PIFu in canonical space for animatable avatar generation!

**[Robust 3D Self-portraits in Seconds (CVPR 2020)](http://www.liuyebin.com/portrait/portrait.html)**  
*Zhe Li, Tao Yu, Chuanyu Pan, Zerong Zheng, Yebin Liu*

They extend PIFu to RGBD + introduce "PIFusion" utilizing PIFu reconstruction for non-rigid fusion.


----------
### For commercial queries, please contact:

Hao Li: hao@hao-li.com ccto: ruilongl@usc.edu

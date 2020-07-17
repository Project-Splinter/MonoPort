
## Install TensorRT docker container
[Instructions](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html)
```
# pull the docker image. 
# Note tensorrt:20.06-py3 comes with CUDA11.0 which is not yet supported by pytorch. So we use tensorrt:20.03-py3
# which has CUDA10.2 built in.
docker pull nvcr.io/nvidia/tensorrt:20.03-py3
# create a container from the image
docker create --network host --gpus all -it -v /media/linux_data:/media/linux_data nvcr.io/nvidia/tensorrt:20.03-py3
# attach to that container
docker start -a -i `docker ps -q -l`
```

Install necessaries inside container:
```
# install pytorch for CUDA10.2
pip install torch torchvision
```

Install this repo:
```
cd /media/linux_data/projects/MonoPort-7.9/
pip install yacs
python setup.py develop
```

[ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt/tree/7.0) supports dynamic input
```
git clone -b 7.0 https://github.com/onnx/onnx-tensorrt.git --recursive
apt-get update
apt-get install libprotobuf-dev protobuf-compiler
cd onnx-tensorrt
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=/workspace/tensorrt 
make -j`proc`
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

<!-- Then install conda:
```
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh -P ./cache
chmod +x ./cache/Anaconda3-2019.10-Linux-x86_64.sh
sh ./cache/Anaconda3-2019.10-Linux-x86_64.sh
``` -->

<!-- Then install pytorch and some python libraries
```
source ~/.bashrc
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install pandas lmdb IPython scipy Pillow numpy tqdm
conda install -c conda-forge jupyterlab
conda install -c conda-forge freeimage
pip install opencv-python
conda install -c conda-forge python-lmdb
``` -->

<!-- 
Then install tensorRT python tools
```
/opt/tensorrt/python/python_setup.sh
``` -->

## Update CUDA to 10.2
[TensorRT guideline](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#overview) indicates it only works with CUDA 10.2 or 11.0RC.



## Install PyCUDA
[Instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda)
```
pip install 'pycuda>=2019.1.1'
```
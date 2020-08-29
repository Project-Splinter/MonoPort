import argparse
import glob
import torch
import tqdm
import numpy as np
import cv2

import streamer_pytorch as streamer
import human_inst_seg

from dataloader import DataLoader
from scene import MonoPortScene
from recon import pifu_calib, forward_vertices


########################################
## load configs
########################################
parser = argparse.ArgumentParser()
parser.add_argument(
    '--camera', action="store_true")
parser.add_argument(
    '--images', default="", nargs="*")
parser.add_argument(
    '--image_folder', default=None)
parser.add_argument(
    '--videos', default="", nargs="*")
parser.add_argument(
    '--loop', action="store_true")
parser.add_argument(
    '--vis', action="store_true")
parser.add_argument(
    '--use_VRweb', action="store_true")
args = parser.parse_args()
device = 'cuda:0'
scaled_boxes = [torch.Tensor([[ 50.0,  0.0, 450.0, 500.0]]).to(device)]


########################################
## initialize data streamer
########################################
print (f'initialize data streamer ...')
if args.camera:
    data_stream = streamer.CaptureStreamer(pad=False)
elif len(args.videos) > 0:
    data_stream = streamer.VideoListStreamer(
        args.videos * (10 if args.loop else 1))
elif len(args.images) > 0:
    data_stream = streamer.ImageListStreamer(
        args.images * (10000 if args.loop else 1))
elif args.image_folder is not None:
    images = sorted(glob.glob(args.image_folder+'/*.jpg'))
    images += sorted(glob.glob(args.image_folder+'/*.png'))
    data_stream = streamer.ImageListStreamer(
        images * (10 if args.loop else 1))


########################################
## human segmentation model
########################################
seg_engine = human_inst_seg.Segmentation(
    device=device, verbose=False)
seg_engine.eval()


processors=[
    lambda data: {"input": data.to(device, non_blocking=True)},

    # instance segmentation:
    lambda data_dict: {
        **data_dict, 
        **dict(zip(
            ["segm", "bboxes", "probs"], 
            seg_engine(data_dict["input"], scaled_boxes)
        ))},
]


########################################
## build async processor
########################################
loader = DataLoader(
    data_stream, 
    batch_size=1, 
    num_workers=1, 
    pin_memory=True,
    processors=processors,
)


@torch.no_grad()
def main_loop():
    for data_dict in tqdm.tqdm(loader):
        segm = data_dict["segm"].cpu().numpy()[0].transpose(1, 2, 0) # [512, 512, 4]
        input = (segm[:, :, 0:3] * 0.5) + 0.5
        output = (segm[:, :, 0:3] * segm[:, :, 3:4] * 0.5) + 0.5
        x1, y1, x2, y2 = scaled_boxes[0].cpu().numpy()[0]

        window = np.hstack([input, output]).astype(np.float32)
        window = np.uint8(window[:, :, ::-1] * 255) # To BGR
        window = cv2.rectangle(cv2.UMat(window), (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

        window = cv2.resize(window, (0, 0), fx=3, fy=3)
        cv2.imshow('segmenation', window)
        cv2.waitKey(1)
        
        
if __name__ == '__main__':
    try:
        main_loop()
    except Exception as e:
        print (e)
        del data_stream
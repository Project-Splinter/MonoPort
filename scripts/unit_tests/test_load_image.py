import glob
import torch
import torchvision
import cv2
import numpy as np

from monoport.lib.dataset.utils import load_image

np.random.seed(9999)

# parsing files
# image_paths = sorted(glob.glob(
#     '/home/rui/local/projects/release/PIFu/data/static/RENDER/*/*.jpg'))
# mask_paths = sorted(glob.glob(
#     '/home/rui/local/projects/release/PIFu/data/static/MASK/*/*.png'))
# assert len(image_paths) == len(mask_paths)
# num_total = len(image_paths)

image_paths = sorted(glob.glob(
    '/media/linux_data/data/pifu_orth_v1/*/*/*/render/359.png'))
mask_paths = [None] * len(image_paths)
assert len(image_paths) == len(mask_paths)
num_total = len(image_paths)

# load
nrow = 8
ncol = 8

idxs = np.random.choice(range(num_total), size=nrow*ncol)
images = []
for idx in idxs:
    image, mask = load_image(
        image_paths[idx], mask_paths[idx], 
        crop_size=480, input_size=512, 
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        blur=1.0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.0)
    images.append(image)
images = torch.stack(images)

# save
torchvision.utils.save_image(
    images, './data/test_load_image.jpg', 
    nrow=nrow, normalize=True, padding=10)

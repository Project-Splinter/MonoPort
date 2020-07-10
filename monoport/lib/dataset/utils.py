import numpy as np
import random
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def load_image(
    image_path, mask_path=None, 
    crop_size=512, input_size=512, 
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    blur=0.0, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
    """
    brightness (float or tuple of python:float (min, max)) 
    – How much to jitter brightness. brightness_factor is chosen uniformly 
    from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
    Should be non negative numbers.

    contrast (float or tuple of python:float (min, max)) 
    – How much to jitter contrast. contrast_factor is chosen uniformly 
    from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. 
    Should be non negative numbers.

    saturation (float or tuple of python:float (min, max)) 
    – How much to jitter saturation. saturation_factor is chosen uniformly 
    from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
    Should be non negative numbers.

    hue (float or tuple of python:float (min, max)) 
    – How much to jitter hue. hue_factor is chosen uniformly 
    from [-hue, hue] or the given [min, max]. 
    Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    image_to_tensor = transforms.Compose([
        transforms.ColorJitter(
            brightness, contrast, saturation, hue),
        transforms.CenterCrop(crop_size),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    mask_to_tensor = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])
    
    if mask_path is not None:
        mask = Image.open(mask_path).split()[-1]
        image = Image.open(image_path).convert('RGB')
    else:
        rgba = Image.open(image_path).convert('RGBA')
        mask = rgba.split()[-1]
        image = rgba.convert('RGB')
    
    if blur > 0:
        radius = np.random.uniform(0, blur)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    image = image_to_tensor(image)
    mask = mask_to_tensor(mask)
    image = image * (mask > 0.5).float()
    return image, mask

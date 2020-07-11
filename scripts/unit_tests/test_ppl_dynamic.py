import numpy as np
import torch
import torchvision
from monoport.lib.common.config import get_cfg_defaults
from monoport.lib.dataset.ppl_dynamic import PPLDynamicDataset
from monoport.lib.dataset.utils import projection

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    dataset = PPLDynamicDataset(
        cfg.dataset, 
        mean=cfg.netG.mean, 
        std=cfg.netG.std,
        training=True)

    data = dataset[0]
    for k, v in data.items():
        if isinstance(v, str) or isinstance(v, int):
            print (k, v)
        else:
            print (k, v.shape)

    # load
    num_total = len(dataset)
    nrow = 4
    ncol = 4

    idxs = np.random.choice(range(num_total), size=nrow*ncol)
    images = []
    for idx in idxs:
        data = dataset[idx]
        image = data['image']
        points = projection(
            data['samples_geo'].numpy(), 
            data['calib'].numpy())
        points = (points * 0.5 + 0.5) * 512
        labels = data['labels_geo'].numpy()
        for p, l in zip(points, labels):
            x, y, z = p
            if l > 0.5:
                image[:, int(y), int(x)] = 1
        images.append(image)
    images = torch.stack(images)

    # save
    torchvision.utils.save_image(
        images, './data/test_ppl_dynamic.jpg', 
        nrow=nrow, normalize=True, padding=10)

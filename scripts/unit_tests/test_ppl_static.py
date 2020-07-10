from yacs.config import CfgNode as CN
from monoport.lib.dataset.ppl_static import PPLStaticDataset
from monoport.lib.common.config import get_cfg_defaults

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    dataset = PPLStaticDataset(
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
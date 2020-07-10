import torch
from yacs.config import CfgNode as CN
from monoport.lib.modeling.MonoPortNet import MonoPortNet

if __name__ == '__main__':
    import tqdm
    torch.manual_seed(9999)

    device = 'cuda:0'
    input = torch.randn(1, 3, 512, 512).to(device)
    points = torch.randn(1, 3, 50000).to(device)
    calibs = torch.randn(1, 3, 4).to(device)

    opt_net = CN()
    opt_net.mean = (0.0, 0.0, 0.0)
    opt_net.std = (1.0, 1.0, 1.0)
    opt_net.projection = "orthogonal" 
    
    # --- netG:backbone options ---
    opt_net.backbone = CN()
    opt_net.backbone.IMF = 'Yolov4Filters'

    # --- netG:normalizer options ---
    opt_net.normalizer = CN()
    opt_net.normalizer.IMF = 'PIFuNomalizer'
    
    # --- netG:head options ---
    opt_net.head = CN()
    opt_net.head.IMF = 'PIFuNetGMLP'

    netG = MonoPortNet(opt_net).to(device)
    netG.load_legacy_pifu('./data/Yolov4/netG_epoch_latest')

    outputs = netG.filter(input)
    for stage, output_stage in enumerate(outputs):
        for lvl, output_lvl in enumerate(output_stage):
            print (f'stage: {stage}, lvl: {lvl}', output_lvl.shape)

    with torch.no_grad():
        outputs = netG(input, points, calibs)

    with torch.no_grad(): # 9.48 fps
        for _ in tqdm.tqdm(range(1000)):
            outputs = netG(input, points, calibs)
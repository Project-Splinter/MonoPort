import torch
from monoport.lib.modeling.MonoPortNet import PIFuNetG

if __name__ == '__main__':
    import tqdm
    torch.manual_seed(9999)

    device = 'cuda:0'
    input = torch.randn(1, 3, 512, 512).to(device)
    points = torch.randn(1, 3, 50000).to(device)
    calibs = torch.randn(1, 3, 4).to(device)

    netG = PIFuNetG().to(device)
    netG.load_legacy_pifu('./data/PIFu/net_G')

    outputs = netG.filter(input)
    for stage, output_stage in enumerate(outputs):
        for lvl, output_lvl in enumerate(output_stage):
            print (f'stage: {stage}, lvl: {lvl}', output_lvl.shape)

    with torch.no_grad():
        outputs = netG(input, points, calibs)

    with torch.no_grad(): # 6.86 fps
        for _ in tqdm.tqdm(range(1000)):
            outputs = netG(input, points, calibs)
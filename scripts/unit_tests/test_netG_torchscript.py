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

    outputs = netG.image_filter(input)
    for stage, output_stage in enumerate(outputs):
        for lvl, output_lvl in enumerate(output_stage):
            print (f'stage: {stage}, lvl: {lvl}', output_lvl.shape)

    # torchscript
    image_filter = torch.jit.trace(netG.image_filter, (input,))
    print (image_filter.code)

    with torch.no_grad(): # 
        for _ in tqdm.tqdm(range(1000)):
            outputs = image_filter(input)
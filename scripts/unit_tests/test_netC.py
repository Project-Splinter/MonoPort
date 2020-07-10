import torch
from monoport.lib.modeling.MonoPortNet import PIFuNetG, PIFuNetC

if __name__ == '__main__':
    import tqdm
    torch.manual_seed(9999)

    device = 'cuda:0'
    input = torch.randn(1, 3, 512, 512).to(device)
    points = torch.randn(1, 3, 50000).to(device)
    calibs = torch.randn(1, 3, 4).to(device)
    
    netG = PIFuNetG().to(device)
    netG.load_legacy_pifu('./data/PIFu/net_G')
    netC = PIFuNetC().to(device)
    netC.load_legacy_pifu('./data/PIFu/net_C')

    with torch.no_grad():
        feat_prior = netG.filter(input)[-1][-1]
        print (feat_prior.shape)

    with torch.no_grad():
        outputs = netC(input, points, calibs, feat_prior=feat_prior)
        print (outputs.shape)

    with torch.no_grad(): # 16.66 fps
        for _ in tqdm.tqdm(range(1000)):
            outputs = netC(input, points, calibs, feat_prior=feat_prior)
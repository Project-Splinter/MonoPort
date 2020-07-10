import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, num_views=1, no_residual=True, last_op=None):
        super(SurfaceClassifier, self).__init__()

        self.filters = nn.ModuleList()
        self.num_views = num_views
        self.no_residual = no_residual
        filter_channels = filter_channels
        self.last_op = last_op

        if self.no_residual:
            for l in range(0, len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                # self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(0, len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                # self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''

        :param feature: list of [BxC_inxHxW] tensors of image features
        :param xy: [Bx3xN] tensor of (x,y) coodinates in the image plane
        :return: [BxC_outxN] tensor of features extracted at the coordinates
        '''

        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = f(y)
            else:
                y = f(
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        if self.last_op:
            y = self.last_op(y)

        return y


def PIFuNetGMLP(*args, **kwargs):
    num_views = 1
    filter_channels = [257, 1024, 512, 256, 128, 1]
    no_residual = False
    last_op = nn.Sigmoid()
    return SurfaceClassifier(filter_channels, num_views, no_residual, last_op)


def PIFuNetCMLP(*args, **kwargs):
    num_views = 1
    filter_channels = [513, 1024, 512, 256, 128, 3]
    no_residual = False
    last_op = nn.Tanh()
    return SurfaceClassifier(filter_channels, num_views, no_residual, last_op)


if __name__ == "__main__":
    import tqdm

    device = 'cuda:0'
    # netG
    input = torch.randn(1, 257, 50000).to(device)
    model = PIFuNetGMLP().to(device)

    with torch.no_grad():
        outputs = model(input)
        print (outputs.shape)

    with torch.no_grad(): # 38.13 fps
        for _ in tqdm.tqdm(range(1000)):
            outputs = model(input)

    # netC
    input = torch.randn(1, 513, 50000).to(device)
    model = PIFuNetCMLP().to(device)

    with torch.no_grad():
        outputs = model(input)
        print (outputs.shape)

    with torch.no_grad(): # 23.71 fps
        for _ in tqdm.tqdm(range(1000)):
            outputs = model(input)

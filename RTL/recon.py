import torch
import numpy as np

@torch.no_grad()
def pifu_calib(extrinsic, intrinsic, device="cuda:0"):
    pifu_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # orthognal
    intrinsic = intrinsic.copy()
    intrinsic[2, 2] = intrinsic[0, 0]
    intrinsic[2, 3] = 0
    extrinsic = extrinsic.copy()
    extrinsic[2, 3] = 0

    calib = np.linalg.inv(
        np.matmul(np.matmul(intrinsic, extrinsic), pifu_matrix))
    
    calib_tensor = torch.from_numpy(calib).unsqueeze(0).float()
    calib_tensor = calib_tensor.to(device)
    return calib_tensor

@torch.no_grad()
def forward_vertices(sdf, direction="front"):
    '''
        - direction: "front" | "back" | "left" | "right"
    '''
    device = sdf.device
    resolution = sdf.size(2)
    if direction == "front":
        pass
    elif direction == "left":
        sdf = sdf.permute(2, 1, 0)
    elif direction == "back":
        inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
        sdf = sdf[inv_idx, :, :]
    elif direction == "right":
        inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
        sdf = sdf[inv_idx, :, :]
        sdf = sdf.permute(2, 1, 0)

    inv_idx = torch.arange(sdf.size(2)-1, -1, -1).long()
    sdf = sdf[inv_idx, :, :]
    sdf_all = sdf.permute(2, 1, 0)

    # shadow
    grad_v = (sdf_all>0.5) * torch.linspace(resolution, 1, steps=resolution).to(device)
    grad_c = torch.ones_like(sdf_all) * torch.linspace(0, resolution-1, steps=resolution).to(device)
    max_v, max_c = grad_v.max(dim=2)
    shadow = grad_c > max_c.view(resolution, resolution, 1)
    keep = (sdf_all>0.5) & (~shadow)
    
    p1 = keep.nonzero().t() #[3, N]
    p2 = p1.clone() # z
    p2[2, :] = (p2[2, :]-2).clamp(0, resolution)
    p3 = p1.clone() # y
    p3[1, :] = (p3[1, :]-2).clamp(0, resolution)
    p4 = p1.clone() # x
    p4[0, :] = (p4[0, :]-2).clamp(0, resolution)

    v1 = sdf_all[p1[0, :], p1[1, :], p1[2, :]]
    v2 = sdf_all[p2[0, :], p2[1, :], p2[2, :]]
    v3 = sdf_all[p3[0, :], p3[1, :], p3[2, :]]
    v4 = sdf_all[p4[0, :], p4[1, :], p4[2, :]]

    X = p1[0, :].long() #[N,]
    Y = p1[1, :].long() #[N,]
    Z = p2[2, :].float() * (0.5 - v1) / (v2 - v1) + p1[2, :].float() * (v2 - 0.5) / (v2 - v1) #[N,]
    Z = Z.clamp(0, resolution)

    # normal
    norm_z = v2 - v1
    norm_y = v3 - v1
    norm_x = v4 - v1
    # print (v2.min(dim=0)[0], v2.max(dim=0)[0], v3.min(dim=0)[0], v3.max(dim=0)[0])

    norm = torch.stack([norm_x, norm_y, norm_z], dim=1)
    norm = norm / torch.norm(norm, p=2, dim=1, keepdim=True)

    return X, Y, Z, norm

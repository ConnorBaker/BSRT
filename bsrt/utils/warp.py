import torch
import torch.nn.functional as F
from torch import Tensor

from bsrt.utils.types import InterpolationType


def warp(feat: Tensor, flow: Tensor, mode: InterpolationType = "bilinear", padding_mode: str = "zeros") -> Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow im1 --> im2

    input flow must be in format (x, y) at every pixel
    feat: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow (x, y)

    """
    _B, _C, H, W = feat.size()
    # print(feat.device, flow.device)

    # mesh grid
    rowv, colv = torch.meshgrid([torch.arange(0.5, H + 0.5), torch.arange(0.5, W + 0.5)], indexing="ij")
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float()
    # print(grid.device, flow.device, feat.device)
    grid = grid + flow

    # scale grid to [-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0

    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)

    grid_norm = grid_norm.permute(0, 2, 3, 1)

    output = F.grid_sample(feat, grid_norm, mode=mode, align_corners=False, padding_mode=padding_mode)

    return output

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.weight_norm import weight_norm


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class WideActResBlock(nn.Module):
    def __init__(self, nf=64):
        super(WideActResBlock, self).__init__()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.8
        kernel_size = 3
        act = nn.ReLU(True)

        body.append(weight_norm(nn.Conv2d(nf, nf * expand, 1, padding=1 // 2)))
        body.append(act)
        body.append(weight_norm(nn.Conv2d(nf * expand, int(nf * linear), 1, padding=1 // 2)))
        body.append(
            weight_norm(nn.Conv2d(int(nf * linear), nf, kernel_size, padding=kernel_size // 2))
        )

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


# TODO: Do we really need gradients for this?
def flow_warp(
    x: Tensor,
    flow: Tensor,
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
) -> Tensor:
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    N, C, H, W = x.size()
    # mesh grid
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W, dtype=x.dtype, device=x.device),
        torch.arange(0, H, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2).requires_grad_(False)  # W(x), H(y), 2
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x, vgrid_scaled, padding_mode=padding_mode, mode="bilinear", align_corners=True
    )
    return output

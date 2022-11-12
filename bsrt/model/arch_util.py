from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


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

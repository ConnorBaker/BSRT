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


def flow_warp(
    x: Tensor,
    flow: Tensor,
    interp_mode="bilinear",
    padding_mode="zeros",
    align_corners=True,
    use_pad_mask=False,
):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel
            dimension. The mask is generated according to the grid_sample results of the padded
            dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2).requires_grad_(False)  # W(x), H(y), 2
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if (
        interp_mode == "nearest4"
    ):  # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(
            x,
            torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output01 = F.grid_sample(
            x,
            torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output10 = F.grid_sample(
            x,
            torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        output11 = F.grid_sample(
            x,
            torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3),
            mode="nearest",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(
            x,
            vgrid_scaled,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


# def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
#     """Warp an image or feature map with optical flow
#     Args:
#         x (Tensor): size (N, C, H, W)
#         flow (Tensor): size (N, H, W, 2), normal value
#         interp_mode (str): 'nearest' or 'bilinear'
#         padding_mode (str): 'zeros' or 'border' or 'reflection'

#     Returns:
#         Tensor: warped image or feature map
#     """
#     assert x.size()[-2:] == flow.size()[1:3]
#     B, C, H, W = x.size()
#     # mesh grid
#     grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
#     grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#     grid.requires_grad = False
#     grid = grid.type_as(x)
#     vgrid = grid + flow
#     # scale grid to [-1,1]
#     vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
#     vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
#     vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#     output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
#     return output

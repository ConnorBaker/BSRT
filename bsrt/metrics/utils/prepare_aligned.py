from typing import Tuple, Union

import torch
import torch.nn as nn
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from utils.bilinear_upsample_2d import bilinear_upsample_2d
from utils.spatial_color_alignment import match_colors
from utils.warp import warp


def prepare_aligned(
    alignment_net: nn.Module,
    pred: Tensor,
    gt: Tensor,
    burst_input: Tensor,
    sr_factor: int,
    kernel_size: int,
    gaussian_kernel: Tensor,
    boundary_ignore: Union[int, None],
) -> Tuple[Tensor, Tensor, Tensor]:
    # Estimate flow between the prediction and the ground truth
    with torch.no_grad():
        flow = alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

    # Warp the prediction to the ground truth coordinates
    pred_warped = warp(pred, flow)

    # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
    # the input and the ground truth
    sr_factor = sr_factor
    ds_factor = 1.0 / float(2.0 * sr_factor)
    flow_ds = (
        bilinear_upsample_2d(
            flow,
            scale_factor=ds_factor,
            recompute_scale_factor=True,
        )
        * ds_factor
    )

    burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
    burst_0_warped = warp(burst_0, flow_ds)
    frame_gt_ds = bilinear_upsample_2d(
        gt,
        scale_factor=ds_factor,
        recompute_scale_factor=True,
    )

    # Match the colorspace between the prediction and ground truth
    pred_warped_m, valid = match_colors(
        frame_gt_ds, burst_0_warped, pred_warped, kernel_size, gaussian_kernel
    )

    # Ignore boundary pixels if specified
    pred_warped_m = ignore_boundary(pred_warped_m, boundary_ignore)
    gt = ignore_boundary(gt, boundary_ignore)
    valid = ignore_boundary(valid, boundary_ignore)

    return pred_warped_m, gt, valid

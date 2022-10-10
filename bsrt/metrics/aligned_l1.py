from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from utils.spatial_color_alignment import get_gaussian_kernel
from metrics.utils.prepare_aligned import prepare_aligned
from torchmetrics.metric import Metric

# TODO: Honestly, this should just be a flag for the L1 class.


class AlignedL1(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(
        self,
        alignment_net: nn.Module,
        sr_factor: int = 4,
        boundary_ignore: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
        # self.gauss_kernel = _gaussian_kernel_2d(11, 1.5)
        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        pred_warped_m, gt, valid = prepare_aligned(
            alignment_net=self.alignment_net,
            pred=pred,
            gt=gt,
            burst_input=burst_input,
            kernel_size=self.ksz,
            gaussian_kernel=self.gauss_kernel,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )

        pred_warped_m = pred_warped_m.contiguous()
        gt = gt.contiguous()
        # Estimate MSE
        mse = F.l1_loss(pred_warped_m, gt, reduction="none")

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)
        self.mse = mse

    def compute(self) -> Tensor:
        return self.mse

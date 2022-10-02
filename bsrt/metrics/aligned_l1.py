from __future__ import annotations
from dataclasses import dataclass, field
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from torchmetrics.metric import Metric
from typing import ClassVar
from utils.spatial_color_alignment import get_gaussian_kernel
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Honestly, this should just be a flag for the L1 class.


@dataclass
class AlignedL1(Metric):
    full_state_update: ClassVar[bool] = False
    alignment_net: nn.Module
    sr_factor: int = 4
    boundary_ignore: int | None = None
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    # Losses
    mse: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
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

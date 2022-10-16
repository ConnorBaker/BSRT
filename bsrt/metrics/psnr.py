from __future__ import annotations

import math
from typing import ClassVar

import torch
from metrics.l2 import L2
from torch import Tensor
from torchmetrics.metric import Metric


# TODO: This serves as a catch-all loss function. We should split it into multiple loss functions, and use metrics provided by torchmetrics where possible.
class PSNR(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    max_value: float = 1.0
    l2: L2

    # Losses
    psnr: Tensor
    ssim: Tensor
    lpips: Tensor

    def __init__(
        self, boundary_ignore: int | None = None, max_value: float = 1.0
    ) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.max_value = max_value
        self.l2 = L2(boundary_ignore=self.boundary_ignore)
        self.add_state("psnr", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Tensor | None = None) -> None:
        """
        Args:
            pred: (B, C, H, W)
            gt: (B, C, H, W)
            valid: (B, C, H, W)
        """
        assert (
            pred.dim() == 4
        ), f"pred must be a 4D tensor, actual shape was {pred.shape}"
        assert (
            pred.shape == gt.shape
        ), f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}"

        mse, ssim, lpips = self.l2(pred, gt, valid=valid)
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()
        self.psnr = psnr
        self.ssim = ssim
        self.lpips = lpips

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.psnr, self.ssim, self.lpips

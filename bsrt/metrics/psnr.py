from __future__ import annotations

import math
from typing import ClassVar

import torch
from metrics.mse import MSE
from torch import Tensor


class PSNR(MSE):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    max_value: float = 1.0

    # Losses
    psnr: Tensor

    def __init__(
        self, boundary_ignore: int | None = None, max_value: float = 1.0
    ) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.max_value = max_value
        self.add_state("psnr", default=torch.tensor(0), dist_reduce_fx="mean")

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
        mse = super()(pred, gt, valid)
        self.psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

    def compute(self) -> Tensor:
        return self.psnr

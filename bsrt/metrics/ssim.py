from __future__ import annotations

from typing import ClassVar

import torch
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric


class SSIM(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None

    # Losses
    ssim: Tensor

    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")

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

        pred = ignore_boundary(pred, self.boundary_ignore).type_as(gt)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        assert pred.device == gt.device and (
            (gt.device == valid.device) if valid is not None else True
        ), f"pred, gt, and valid must be on the same device"

        self.ssim: Tensor = compute_ssim(
            pred.contiguous(),
            gt.contiguous(),
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
            data_range=1.0,
        )  # type: ignore

    def compute(self) -> Tensor:
        return self.ssim

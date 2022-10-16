from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F
from metrics.utils.ignore_boundry import ignore_boundary
from torch import Tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.metric import Metric


class L2(Metric):
    full_state_update: ClassVar[bool] = False
    boundary_ignore: int | None = None
    # FIXME: Why does LPIPS have unused model parameters when lpips=True (the default setting)?
    loss_fn: LPIPS

    # Losses
    lpips: Tensor

    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.loss_fn = LPIPS(net="alex", lpips=True, normalize=True)
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

        pred = ignore_boundary(pred, self.boundary_ignore).type_as(gt)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        assert pred.device == gt.device and (
            (gt.device == valid.device) if valid is not None else True
        ), f"pred, gt, and valid must be on the same device"

        self.lpips = self.loss_fn(
            pred.contiguous(),
            gt.contiguous(),
        )

    def compute(self) -> Tensor:
        return self.lpips

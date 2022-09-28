from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.metric import Metric
from metrics.utils.ignore_boundry import ignore_boundary


class L1(Metric):
    full_state_update = False

    def __init__(self, boundary_ignore: Optional[int] = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, valid: Optional[Tensor] = None) -> None:
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

        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        mse: Tensor = torch.tensor(0.0)
        acc: Tensor = torch.tensor(0.0)
        for pred, gt, valid in zip(
            pred, gt, valid if valid is not None else [None] * len(pred)
        ):
            if valid is None:
                mse = F.l1_loss(pred, gt)
            else:
                mse_tensor: Tensor = F.l1_loss(pred, gt, reduction="none")

                eps: float = 1e-12
                elem_ratio: float = mse_tensor.numel() / valid.numel()
                # TODO: Why is it necessary to cast to float?
                mse = (mse_tensor * valid.float()).sum() / (
                    valid.float().sum() * elem_ratio + eps
                )

            acc += mse

        self.mse = acc / len(pred)

    def compute(self) -> Tensor:
        return self.mse

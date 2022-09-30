from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.metric import Metric
from metrics.utils.ignore_boundry import ignore_boundary


class L1(Metric):
    full_state_update = False

    def __init__(self, boundary_ignore: int | None = None) -> None:
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")

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

        pred = ignore_boundary(pred, self.boundary_ignore)
        gt = ignore_boundary(gt, self.boundary_ignore)
        valid = ignore_boundary(valid, self.boundary_ignore)

        assert pred.device == gt.device and (
            (gt.device == valid.device) if valid is not None else True
        ), f"pred, gt, and valid must be on the same device"

        acc: Tensor = torch.tensor(0.0, device=pred.device)
        for pred, gt, valid in zip(
            pred, gt, valid if valid is not None else [None] * len(pred)
        ):
            if valid is None:
                mse: Tensor = F.l1_loss(pred, gt)
            else:
                mse_tensor: Tensor = F.l1_loss(pred, gt, reduction="none")

                eps: float = 1e-12
                elem_ratio: float = mse_tensor.numel() / valid.numel()
                # TODO: Why is it necessary to cast to float?
                mse: Tensor = (mse_tensor * valid.float()).sum() / (
                    valid.float().sum() * elem_ratio + eps
                )

            acc += mse

        self.mse = acc / len(pred)

    def compute(self) -> Tensor:
        return self.mse

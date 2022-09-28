from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor
from typing import Tuple
from metrics.utils.compute_psnr import compute_psnr
from metrics.l2 import L2
from torchmetrics.metric import Metric


class PSNR(Metric):
    full_state_update = False

    def __init__(
        self, boundary_ignore: Optional[int] = None, max_value: float = 1.0
    ) -> None:
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value
        self.add_state("psnr", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

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
        all_scores = [
            compute_psnr(
                self.l2,
                self.max_value,
                p.unsqueeze(0),
                g.unsqueeze(0),
                v.unsqueeze(0) if v is not None else None,
            )
            for p, g, v in zip(
                pred, gt, valid if valid is not None else [None] * len(pred)
            )
        ]
        # psnr, ss, lp = sum(psnr_all) / len(psnr_all)
        self.psnr: Tensor = sum([score[0] for score in all_scores]) / len(all_scores)  # type: ignore
        self.ssim: Tensor = sum([score[1] for score in all_scores]) / len(all_scores)  # type: ignore
        self.lpips: Tensor = sum([score[2] for score in all_scores]) / len(all_scores)  # type: ignore

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.psnr, self.ssim, self.lpips

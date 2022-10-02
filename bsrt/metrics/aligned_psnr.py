from __future__ import annotations
from dataclasses import dataclass, field
from metrics.aligned_l2 import AlignedL2
from metrics.utils.compute_psnr import compute_psnr
from torch import Tensor
from torchmetrics.metric import Metric
from typing import ClassVar
import torch


@dataclass
class AlignedPSNR(Metric):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    sr_factor: int = 4
    boundary_ignore: int | None = None
    max_value: float = 1.0
    l2: AlignedL2 = field(init=False)

    # Losses
    psnr: Tensor = field(init=False)
    ssim: Tensor = field(init=False)
    lpips: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.l2 = AlignedL2(
            alignment_net=self.alignment_net,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )
        self.add_state("psnr", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        all_scores = [
            compute_psnr(
                self.l2, self.max_value, p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0)
            )
            for p, g, bi in zip(pred, gt, burst_input)
        ]
        self.psnr: Tensor = sum([score[0] for score in all_scores]) / len(all_scores)  # type: ignore
        self.ssim: Tensor = sum([score[1] for score in all_scores]) / len(all_scores)  # type: ignore
        self.lpips: Tensor = sum([score[2] for score in all_scores]) / len(all_scores)  # type: ignore

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.psnr, self.ssim, self.lpips

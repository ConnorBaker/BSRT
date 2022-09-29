from __future__ import annotations
import torch
from torch import Tensor
from metrics.aligned_l2 import AlignedL2
from metrics.utils.compute_psnr import compute_psnr
from torchmetrics.metric import Metric


class AlignedPSNR(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = AlignedL2(
            alignment_net=alignment_net,
            sr_factor=sr_factor,
            boundary_ignore=boundary_ignore,
        )
        self.max_value = max_value
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

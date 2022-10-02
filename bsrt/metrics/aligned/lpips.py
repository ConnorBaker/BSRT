from __future__ import annotations
from dataclasses import dataclass, field
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from typing import ClassVar
from utils.spatial_color_alignment import get_gaussian_kernel
import torch


@dataclass
class AlignedLPIPS(Metric):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    sr_factor: int = 4
    boundary_ignore: int | None = None
    loss_fn: LPIPS = field(init=False, default_factory=lambda: LPIPS(net="alex"))
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    # Losses
    lpips: Tensor = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)
        self.add_state("lpips", default=torch.tensor(0), dist_reduce_fx="mean")

    def _lpips(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> Tensor:  # type: ignore
        pred_warped_m, gt, _valid = prepare_aligned(
            alignment_net=self.alignment_net,
            pred=pred,
            gt=gt,
            burst_input=burst_input,
            kernel_size=self.ksz,
            gaussian_kernel=self.gauss_kernel,
            sr_factor=self.sr_factor,
            boundary_ignore=self.boundary_ignore,
        )

        # Estimate MSE
        mse = self.loss_fn(pred_warped_m.contiguous(), gt.contiguous()).squeeze()
        return mse

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        lpips_all = [
            self._lpips(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0))
            for p, g, bi in zip(pred, gt, burst_input)
        ]
        self.lpips: Tensor = sum(lpips_all) / len(lpips_all)  # type: ignore

    def compute(self) -> Tensor:
        return self.lpips

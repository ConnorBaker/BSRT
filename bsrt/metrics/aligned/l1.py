from __future__ import annotations
from dataclasses import dataclass, field
from metrics.l1 import L1
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from typing import ClassVar
from utils.spatial_color_alignment import get_gaussian_kernel
import torch


@dataclass
class AlignedL1(L1):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    boundary_ignore: int | None = None
    sr_factor: int = 4
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    def __post_init__(self) -> None:
        super().__init__(boundary_ignore=self.boundary_ignore)
        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        """
        Args:
            pred: (B, C, H, W)
            gt: (B, C, H, W)
            burst_input: (N, B, C, H, W)
        """
        pred_warped_ms: list[Tensor] = []
        gts: list[Tensor] = []
        valids: list[Tensor] = []
        for pred, gt, burst_input in zip(pred, gt, burst_input):
            pred_warped_m, gt, valid = prepare_aligned(
                alignment_net=self.alignment_net,
                pred=pred,
                gt=gt,
                burst_input=burst_input,
                kernel_size=self.ksz,
                gaussian_kernel=self.gauss_kernel,
                sr_factor=self.sr_factor,
                boundary_ignore=self.boundary_ignore,
            )
            pred_warped_ms.append(pred_warped_m)
            gts.append(gt)
            valids.append(valid)

        pred_warped_m = torch.stack(pred_warped_ms)
        gt = torch.stack(gts)
        valid = torch.stack(valids)

        super().update(pred_warped_m, gt, valid)

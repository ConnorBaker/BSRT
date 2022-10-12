from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Union

import torch
from metrics.l1 import L1
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from utils.spatial_color_alignment import get_gaussian_kernel


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False, init=False, kw_only=True)
class AlignedL1(L1):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    boundary_ignore: Union[int, None] = None
    sr_factor: int = 4
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    # TODO: We cannot use the generated init with nn.Modules arguments because we must call the super init before we can call the module init.
    def __init__(
        self,
        alignment_net: torch.nn.Module,
        boundary_ignore: Union[int, None] = None,
        sr_factor: int = 4,
    ) -> None:
        super().__init__(boundary_ignore=boundary_ignore)
        self.alignment_net = alignment_net
        self.boundary_ignore = boundary_ignore
        self.sr_factor = sr_factor
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

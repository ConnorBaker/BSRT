from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Union

import torch
import utils.spatial_color_alignment as sca_utils
from metrics.utils.prepare_aligned import prepare_aligned
from torch import Tensor
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric


# TODO: Using the derivied equals overwrites the default hash method, which we want to inherit from Metric.
@dataclass(eq=False, init=False, kw_only=True)
class AlignedSSIM(Metric):
    full_state_update: ClassVar[bool] = False
    alignment_net: torch.nn.Module
    boundary_ignore: Union[int, None] = None
    sr_factor: int = 4
    gauss_kernel: Tensor = field(init=False)
    ksz: int = field(init=False)

    # Losses
    ssim: Tensor = field(init=False)

    # TODO: We cannot use the generated init with nn.Modules arguments because we must call the super init before we can call the module init.
    def __init__(
        self,
        alignment_net: torch.nn.Module,
        boundary_ignore: Union[int, None] = None,
        sr_factor: int = 4,
    ) -> None:
        super().__init__()
        self.alignment_net = alignment_net
        self.boundary_ignore = boundary_ignore
        self.sr_factor = sr_factor
        self.gauss_kernel, self.ksz = sca_utils.get_gaussian_kernel(sd=1.5)
        self.add_state("ssim", default=torch.tensor(0), dist_reduce_fx="mean")

    def _ssim(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> Tensor:
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

        # Estimate MSE
        # TODO: The generated superresolution image regularly has a range greater than 1.0. Is this a problem?
        mse: Tensor = compute_ssim(
            pred_warped_m.type_as(gt).contiguous(),
            gt.contiguous(),
            gaussian_kernel=True,
            kernel_size=11,
            sigma=1.5,
            reduction="elementwise_mean",
            data_range=1.0,
        )  # type: ignore

        return mse

    def update(self, pred: Tensor, gt: Tensor, burst_input: Tensor) -> None:
        ssim_all = [
            self._ssim(p.unsqueeze(0), g.unsqueeze(0), bi.unsqueeze(0))
            for p, g, bi in zip(pred, gt, burst_input)
        ]
        self.ssim: Tensor = sum(ssim_all) / len(ssim_all)  # type: ignore

    def compute(self) -> Tensor:
        return self.ssim

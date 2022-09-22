from __future__ import annotations
import torch
from torch import Tensor
import utils.spatial_color_alignment as sca_utils
from metrics.utils.prepare_aligned import prepare_aligned

# NOTE: We specifically do not use the LPIPS module torchmetrics ships with since it requires that all inputs are in the range [-1,1] and our SR outputs during training are regularly greater than one.
from torchmetrics.functional.image.ssim import (
    structural_similarity_index_measure as compute_ssim,
)
from torchmetrics.metric import Metric


class AlignedSSIM(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
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

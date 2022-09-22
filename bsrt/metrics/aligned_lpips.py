from __future__ import annotations
import torch
from torch import Tensor
from utils.spatial_color_alignment import get_gaussian_kernel
from metrics.utils.prepare_aligned import prepare_aligned

# NOTE: We specifically do not use the LPIPS module torchmetrics ships with since it requires that all inputs are in the range [-1,1] and our SR outputs during training are regularly greater than one.
from torchmetrics.metric import Metric
from lpips import LPIPS


class AlignedLPIPS(Metric):
    # TODO: See if we need the full metric state (the property full_state_update=True)
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sr_factor = sr_factor
        self.boundary_ignore = boundary_ignore
        self.alignment_net = alignment_net
        self.loss_fn = LPIPS(net="alex")
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

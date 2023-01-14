from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from mfsr_utils.pipelines.camera import demosaic
from mfsr_utils.pipelines.synthetic_burst_generator import SyntheticBurstGeneratorData
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn
from torch._inductor import config
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM

from bsrt.model.bsrt import BSRT
from bsrt.tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from bsrt.tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from bsrt.tuning.lr_scheduler.one_cycle_lr import OneCycleLRParams
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from bsrt.tuning.lr_scheduler.utilities import configure_scheduler
from bsrt.tuning.model.bsrt import BSRTParams
from bsrt.tuning.optimizer.adamw import AdamWParams
from bsrt.tuning.optimizer.sgd import SGDParams
from bsrt.tuning.optimizer.utilities import configure_optimizer

# https://github.com/pytorch/torchdynamo/issues/1967
config.compile_threads = 1

# dead code elimination
config.dce = True  # default is False

# put correctness assertions in generated code
config.size_asserts = False  # default is True

# fuse pointwise into templates
config.epilogue_fusion = True  # default is False
# do epilogue fusions before other fusions
config.epilogue_fusion_first = True  # default is False

# control store vs recompute heuristic
# For fanouts, rematearialization can lead to exponential blowup. So, have
# smaller threshold
# config.realize_reads_threshold = 4
# config.realize_bytes_threshold = 2000
# Threshold to prevent excessive accumulation of ops in one buffer during lowering
# config.realize_acc_reads_threshold = 8

# how many nodes to allow into a single fusion
config.max_fusion_size = 256  # default is 64

# replace small reductions with pointwise, disable with `= 1`
config.unroll_reductions_threshold = 48  # default is 8
# Values of 50 and above cause OOM errors

# Fx-based linear/matmul/bmm + permute/transpose vertical fusion
config.permute_fusion = (
    True  # default is os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"
)

# config specific to codegen/triton.py
# Use cudagraphs on output code
config.triton.cudagraphs = False  # default is True"


@dataclass(eq=False)
class LightningBSRT(LightningModule):
    bsrt_params: BSRTParams
    optimizer_params: AdamWParams | SGDParams
    scheduler_params: (
        CosineAnnealingWarmRestartsParams
        | ExponentialLRParams
        | OneCycleLRParams
        | ReduceLROnPlateauParams
    )

    model: nn.Module = field(init=False)
    train_metrics: MetricCollection = field(init=False)
    valid_metrics: MetricCollection = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        # Initialize model
        model = BSRT(**self.bsrt_params.__dict__)
        self.model = torch.compile(model, fullgraph=False, dynamic=False)
        # Initialize loss functions
        metrics = MetricCollection(
            {
                "psnr": PSNR(data_range=1.0),
                "ms_ssim": MS_SSIM(data_range=1.0),
                "lpips": LPIPS(net_type="alex", normalize=True).requires_grad_(False),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")

        # Mypy thinks save_hyperparameters is a Tensor
        self.save_hyperparameters(  # type: ignore[operator]
            ignore=["model", "train_metrics", "valid_metrics"]
        )

    def forward(self, bursts: Tensor) -> Tensor:  # type: ignore[override]
        ret: Tensor = self.model(bursts)
        return ret

    def training_step(  # type: ignore[override]
        self, batch: SyntheticBurstGeneratorData
    ) -> dict[str, Tensor]:
        bursts = batch["burst"]
        gts = batch["gt"]
        srs: Tensor = self(bursts)

        # Some metrics don't work with bfloat16, so we cast to float32
        gts = gts.to(torch.float32)
        srs = srs.to(torch.float32)

        # Calculate losses
        loss: dict[str, Tensor] = self.train_metrics(srs, gts)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore

        # PyTorch Lightning requires that when validation_step returns a dict, it must contain a
        # key named loss
        loss["loss"] = loss["train/lpips"]
        return loss

    def validation_step(  # type: ignore[override]
        self, batch: SyntheticBurstGeneratorData, batch_idx: int
    ) -> dict[str, Tensor]:
        bursts = batch["burst"]
        gts = batch["gt"]
        srs: Tensor = self(bursts)

        # Some metrics don't work with bfloat16, so we cast to float32
        gts = gts.to(torch.float32)
        srs = srs.to(torch.float32)

        # Calculate losses
        loss: dict[str, Tensor] = self.valid_metrics(srs, gts)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True)  # type: ignore

        # Log the image only for the first batch
        # TODO: We could log different images with different names
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            # Gross hack to work around "RuntimeError: "upsample_nearest2d_out_frame" not
            # implemented for 'BFloat16'"
            orig_dtype = bursts.dtype
            if orig_dtype == torch.bfloat16:
                bursts = bursts.to(torch.float32)

            # TODO: Pyright complains that the type of F.interpolate is partially unknown
            nn_busrts: Tensor = F.interpolate(  # type: ignore
                input=demosaic(bursts[:, 0, :, :]),
                size=None,
                align_corners=None,
                recompute_scale_factor=None,
                antialias=False,
                scale_factor=4,
                # mode="nearest-exact",
            )

            if orig_dtype == torch.bfloat16:
                nn_busrts = nn_busrts.to(bursts.dtype)

            for i, (nn_burst, sr, gt) in enumerate(zip(nn_busrts, srs, gts)):
                # TODO: Pyright complains that the type of log_image is partially unknown
                self.logger.log_image(  # type: ignore
                    key=f"val/sample/{i}",
                    images=[nn_burst, sr, gt],
                    caption=["LR", "SR", "GT"],
                )

        # PyTorch Lightning requires that when validation_step returns a dict, it must contain a
        # key named loss
        loss["loss"] = loss["val/lpips"]
        return loss

    def configure_optimizers(self) -> dict[str, str | Optimizer | LRScheduler]:
        if (
            isinstance(self.scheduler_params, OneCycleLRParams)
            and self.scheduler_params.total_steps is None
        ):
            total_steps = self.trainer.estimated_stepping_batches
            assert (
                isinstance(total_steps, int) and total_steps >= 1
            ), "Cannot use OneCycleLR with infinite or negative total_steps"
            self.scheduler_params.total_steps = total_steps

        opt = configure_optimizer(self.model, self.optimizer_params)
        scheduler = configure_scheduler(opt, self.scheduler_params)

        ret: dict[str, str | Optimizer | LRScheduler] = {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }
        if isinstance(scheduler, ReduceLROnPlateau):
            ret["monitor"] = "val/lpips"

        return ret

    # Set gradients to `None` instead of zero to improve performance.
    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    # Due to the LRScheduler refactor in PyTorch 1.14, we need to override this method to avoid
    # the following error we get with any scheduler:
    #
    # lightning_lite.utilities.exceptions.MisconfigurationException: The provided lr scheduler
    # `OneCycleLR` doesn't follow PyTorch's LRScheduler API. You should override the
    # `LightningModule.lr_scheduler_step` hook with your own logic if you are using a custom LR
    # scheduler.
    def lr_scheduler_step(  # type: ignore[override]
        self, scheduler: LRScheduler, optimizer_idx: int, metric: Any
    ) -> None:
        super().lr_scheduler_step(scheduler, optimizer_idx, metric)

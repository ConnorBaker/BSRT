from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from mfsr_utils.pipelines.camera import demosaic
from mfsr_utils.pipelines.synthetic_burst_generator import SyntheticBurstGeneratorData
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn

if torch.__version__ < "2.0.0":
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
else:
    from torch.optim.lr_scheduler import LRScheduler

from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    validation_metrics: MetricCollection = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()

        # Initialize model
        self.model = BSRT(**self.bsrt_params.__dict__)

        # Initialize loss functions
        metrics = MetricCollection(
            {
                "psnr": PSNR(data_range=1.0),
                "ms_ssim": MS_SSIM(data_range=1.0),
                "lpips": LPIPS(net_type="alex", normalize=True).requires_grad_(False),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.validation_metrics = metrics.clone(prefix="val/")

        # Mypy thinks save_hyperparameters is a Tensor
        self.save_hyperparameters(  # type: ignore[operator]
            ignore=["model", "train_metrics", "validation_metrics"]
        )

    def forward(self, bursts: Tensor) -> Tensor:  # type: ignore[override]
        ret: Tensor = self.model(bursts)
        return ret

    def training_step(  # type: ignore[override]
        self, batch: SyntheticBurstGeneratorData, batch_idx: int
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
        loss["loss"] = abs(1 - loss["train/ms_ssim"])
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
        loss: dict[str, Tensor] = self.validation_metrics(srs, gts)
        self.log_dict(self.validation_metrics, on_step=False, on_epoch=True)  # type: ignore

        # Log the image only for the first batch
        # TODO: We could log different images with different names
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
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

            for i, (nn_burst, sr, gt) in enumerate(zip(nn_busrts, srs, gts)):
                # TODO: Pyright complains that the type of log_image is partially unknown
                self.logger.log_image(  # type: ignore
                    key=f"val/sample/{i}",
                    images=[nn_burst, sr, gt],
                    caption=["LR", "SR", "GT"],
                )

        # PyTorch Lightning requires that when validation_step returns a dict, it must contain a
        # key named loss
        loss["loss"] = abs(1 - loss["val/ms_ssim"])
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
            ret["monitor"] = "val/ms_ssim"

        return ret

    # Set gradients to `None` instead of zero to improve performance.
    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

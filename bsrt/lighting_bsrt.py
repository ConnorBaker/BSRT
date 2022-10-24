from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)

from .data_processing.camera_pipeline import demosaic
from .datasets.synthetic_burst.train_dataset import TrainData
from .model.bsrt import BSRT
from .tuning.lr_scheduler.cosine_annealing_warm_restarts import (
    CosineAnnealingWarmRestartsParams,
)
from .tuning.lr_scheduler.exponential_lr import ExponentialLRParams
from .tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from .tuning.lr_scheduler.utilities import configure_scheduler
from .tuning.model.bsrt import BSRTParams
from .tuning.optimizer.decoupled_adamw import DecoupledAdamWParams
from .tuning.optimizer.decoupled_sgdw import DecoupledSGDWParams
from .tuning.optimizer.utilities import configure_optimizer


@dataclass(eq=False)
class LightningBSRT(LightningModule):
    bsrt_params: BSRTParams
    optimizer_params: Union[DecoupledAdamWParams, DecoupledSGDWParams]
    scheduler_params: Union[
        CosineAnnealingWarmRestartsParams, ExponentialLRParams, ReduceLROnPlateauParams
    ]
    # If use_opts is true, we need composer
    use_speed_opts: bool = False
    use_quality_opts: bool = False

    # lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    model: nn.Module = field(init=False)
    train_metrics: MetricCollection = field(init=False)
    valid_metrics: MetricCollection = field(init=False)

    def __post_init__(self):
        super().__init__()

        # Initialize model
        self.model = BSRT(**self.bsrt_params.__dict__)
        if self.use_quality_opts:
            import composer.functional as cf

            self.model = cf.apply_blurpool(
                self.model,
                replace_convs=True,
                replace_maxpools=True,
                blur_first=True,
                min_channels=16,
            )
            self.model = cf.apply_squeeze_excite(
                self.model, min_channels=128, latent_channels=64
            )
        if self.use_speed_opts:
            import composer.functional as cf

            cf.apply_channels_last(self.model)

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

    def forward(self, bursts: Tensor) -> Tensor:
        return self.model(bursts)

    def training_step(self, batch: TrainData, batch_idx: int) -> Dict[str, Tensor]:
        bursts = batch["burst"]
        gts = batch["gt"]
        srs = self(bursts)

        # Calculate losses
        loss: Dict[str, Tensor] = self.train_metrics(srs, gts)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)  # type: ignore

        # PyTorch Lightning requires that when validation_step returns a dict, it must contain a key named loss
        loss["loss"] = loss["train/lpips"]
        return loss

    def validation_step(self, batch: TrainData, batch_idx: int) -> Dict[str, Tensor]:
        bursts = batch["burst"]
        gts = batch["gt"]
        srs = self(bursts)

        # Calculate losses
        loss: Dict[str, Tensor] = self.valid_metrics(srs, gts)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True)  # type: ignore

        # Log the image only for the first batch
        # TODO: We could log different images with different names
        if batch_idx == 0 and isinstance(self.logger, WandbLogger):
            # Gross hack to work around "RuntimeError: "upsample_nearest2d_out_frame" not implemented for 'BFloat16'"
            nn_busrts: Tensor = F.interpolate(
                demosaic(
                    bursts[:, 0, :, :].to(
                        torch.float32
                        if bursts.dtype == torch.bfloat16
                        else bursts.dtype
                    )
                ),
                scale_factor=4,
                mode="nearest-exact",
            ).to(bursts.dtype)

            for i, (nn_burst, sr, gt) in enumerate(zip(nn_busrts, srs, gts)):
                self.logger.log_image(
                    key=f"val/sample_{i}",
                    images=[nn_burst, sr, gt],
                    caption=["LR", "SR", "GT"],
                )

        # PyTorch Lightning requires that when validation_step returns a dict, it must contain a key named loss
        loss["loss"] = loss["val/lpips"]
        return loss

    def configure_optimizers(self) -> Tuple[Optimizer, _LRScheduler]:
        opt = configure_optimizer(self.model, self.optimizer_params)
        if self.use_speed_opts:
            import composer.functional as cf

            cf.apply_fused_layernorm(self.model, optimizers=opt)

        scheduler = configure_scheduler(opt, self.scheduler_params)

        return opt, scheduler

from pathlib import Path

import torch
from mfsr_utils.datasets.zurich_raw2rgb import ZurichRaw2Rgb
from mfsr_utils.pipelines.synthetic_burst_generator import (
    SyntheticBurstGeneratorData,
    SyntheticBurstGeneratorTransform,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.trainer import Trainer
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from bsrt.lightning_bsrt import LightningBSRT
from bsrt.tuning.lr_scheduler.reduce_lr_on_plateau import ReduceLROnPlateauParams
from bsrt.tuning.model.bsrt import BSRTParams
from bsrt.tuning.optimizer.adamw import AdamWParams

if __name__ == "__main__":
    import os

    os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

    import torch.backends.cuda
    import torch.backends.cudnn

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Desired batch size
    target_batch_size: int = 128

    # Number of batches a single GPU can handle in memory
    single_gpu_batch_size: int = 8

    # Number of GPUs
    num_gpus: int = torch.cuda.device_count()

    # Number of batches to accumulate before performing a backward pass
    actual_batch_size: int = single_gpu_batch_size * num_gpus

    # Number of batches to accumulate before performing a backward pass
    accumulate_batch_size: int = target_batch_size // actual_batch_size

    # Num CPUs
    num_cpus: None | int = os.cpu_count()
    assert num_cpus is not None

    model = LightningBSRT(
        bsrt_params=BSRTParams(
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            drop_rate=0.0,
            mlp_ratio=4,
            flow_alignment_groups=1,
            num_features=64,
            qkv_bias=True,
        ),
        optimizer_params=AdamWParams(
            lr=0.0001,
            beta_gradient=0.9,
            beta_square=0.999,
            eps=1e-08,
            weight_decay=0.01,
        ),
        scheduler_params=ReduceLROnPlateauParams(
            factor=0.1,
        ),
    )

    data_dir = Path("/home/connorbaker/datasets")
    ZurichRaw2Rgb.download_to(data_dir)

    full_dataset: ZurichRaw2Rgb[SyntheticBurstGeneratorData] = ZurichRaw2Rgb(
        data_dir,
        transform=SyntheticBurstGeneratorTransform(
            burst_size=14, crop_sz=256, dtype=torch.float32
        ),
    )
    train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])
    train_data_loader: DataLoader[SyntheticBurstGeneratorData] = DataLoader(
        train_dataset,
        batch_size=single_gpu_batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data_loader: DataLoader[SyntheticBurstGeneratorData] = DataLoader(
        val_dataset,
        batch_size=single_gpu_batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        "group": "testrun",
        "dir": None,
        "reinit": True,
    }

    logger: WandbLogger = WandbLogger(**wandb_kwargs)
    logger.log_hyperparams(model.hparams)  # type: ignore
    logger.watch(model, log="all", log_graph=True)

    trainer = Trainer(
        enable_checkpointing=True,
        gradient_clip_val=0.5,
        accelerator="auto",
        # Static graph must be false for torch.compile to work.
        strategy=DDPSpawnStrategy(static_graph=False, find_unused_parameters=False),
        accumulate_grad_batches=accumulate_batch_size,
        precision=32,
        deterministic=False,
        detect_anomaly=False,
        logger=logger,
    )

    trainer.fit(  # type: ignore
        model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader
    )

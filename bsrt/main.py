from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from .datasets.synthetic_zurich_raw2rgb_data_module import (
    SyntheticZurichRaw2RgbDataModule,
)
from .model.bsrt import BSRT

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

    wandb_kwargs = {
        "entity": "connorbaker",
        "project": "bsrt",
        "group": "16-hyperparameter-tuning-6",
        "dir": None,
        "reinit": True,
    }

    cli = LightningCLI(
        BSRT,
        SyntheticZurichRaw2RgbDataModule,
        run=False,
        save_config_callback=None,
    )

    cli.instantiate_classes()
    model = cli.model
    trainer = cli.trainer
    logger = trainer.logger = WandbLogger(**wandb_kwargs)
    assert isinstance(logger, WandbLogger), "Logger should be set to the WandbLogger"

    logger.log_hyperparams(model.hparams)
    logger.watch(model, log="all", log_graph=True)
    trainer.fit(model, datamodule=cli.datamodule)

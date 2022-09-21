import os
from pathlib import Path
import model.bsrt as bsrt
import loss
from option import Config, config
import pytorch_lightning as pl
from data_modules.zurich_raw2rgb_data_module import ZurichRaw2RgbDataModule


def main():
    pl.seed_everything(0)

    num_workers = os.cpu_count()
    num_workers = num_workers // 2 if num_workers is not None else 0

    _model = bsrt.make_model(config)
    data_module = ZurichRaw2RgbDataModule(
        Path(config.data_dir),
        config.burst_size,
        config.patch_size,
        config.batch_size,
        num_workers,
    )
    trainer = pl.Trainer(accelerator="gpu", devices=-1, strategy="ddp")
    trainer.fit(_model, datamodule=data_module)


if __name__ == "__main__":
    main()

import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import model.bsrt as bsrt
import loss
from option import Config, config
import pytorch_lightning as pl
from datasets.burstsr_dataset import BurstSRDataset
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.synthetic_burst_val_set import SyntheticBurstVal
from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGB



def main():
    pl.seed_everything(0)
    pl.

    batch_size: int = config.batch_size
    if config.data_type == "synthetic":
        train_zurich_raw2rgb = ZurichRAW2RGB(root=config.root, split="train")
        train_data = SyntheticBurst(
            train_zurich_raw2rgb,
            burst_size=config.burst_size,
            crop_sz=config.patch_size,
        )

        # valid_zurich_raw2rgb = ZurichRAW2RGB(root=args.root, split='test')
        # valid_data = SyntheticBurst(valid_zurich_raw2rgb, burst_size=14, crop_sz=1024)
        valid_data = SyntheticBurstVal(root=config.val_root)
    elif config.data_type == "real":
        train_data = BurstSRDataset(
            root=config.root,
            burst_size=config.burst_size,
            crop_sz=config.patch_size,
            random_flip=True,
            center_crop=True,
            split="train",
        )
        valid_data = BurstSRDataset(
            root=config.root, burst_size=14, crop_sz=80, split="val"
        )

    print(f"train data: {len(train_data)}, test data: {len(valid_data)}")

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(
    #     valid_data, shuffle=False
    # )

    num_workers = os.cpu_count()
    num_workers = num_workers // 2 if num_workers is not None else 0
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        # sampler=train_sampler,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        # sampler=valid_sampler,
    )

    _model = bsrt.make_model(config)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(_model, train_loader, valid_loader)
    _loss = loss.Loss(config)


if __name__ == "__main__":
    main()

from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset


class ZurichRaw2RgbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        burst_size: int,
        patch_size: int,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.burst_size = burst_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        dir = self.data_dir / ZurichRaw2RgbDataset.dirname
        if not dir.exists():
            file = self.data_dir / ZurichRaw2RgbDataset.filename
            if not file.exists():
                urls = [ZurichRaw2RgbDataset.url] + ZurichRaw2RgbDataset.mirrors
                while not len(urls) == 0:
                    url = urls.pop()
                    try:
                        download_and_extract_archive(
                            url,
                            download_root=self.data_dir.as_posix(),
                            extract_root=dir.as_posix(),
                            filename=ZurichRaw2RgbDataset.filename,
                        )
                        break
                    except:
                        print(f"Failed to download from {url}")
                        pass
                if not file.exists():
                    raise Exception("Could not download dataset")
            else:
                extract_archive(from_path=file.as_posix(), to_path=dir.as_posix())

    def setup(self, stage=None):
        # Split the training dataset into train and validation
        zrr_full = ZurichRaw2RgbDataset(self.data_dir / ZurichRaw2RgbDataset.dirname)

        full_len = len(zrr_full)
        train_len = int(full_len * 0.75)
        val_len = full_len - train_len

        zrr_train, zrr_val = random_split(zrr_full, [train_len, val_len])

        self.zrr_train = SyntheticBurst(
            zrr_train,
            burst_size=self.burst_size,
            crop_sz=self.patch_size,
        )
        self.zrr_val = SyntheticBurst(
            zrr_val,
            burst_size=self.burst_size,
            crop_sz=self.patch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.zrr_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.zrr_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

import os
from pathlib import Path
import pytorch_lightning as pl
import requests
from zipfile import ZipFile
from torchvision.datasets.folder import VisionDataset
from typing import Callable, Optional
from torch.utils.data import random_split

from torchvision.datasets import MNIST

from torchvision.datasets.utils import download_and_extract_archive


class ZurichRaw2RgbDataset(VisionDataset):
    """ Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename = "zurich-raw-to-rgb.zip"
    dirname = "zurich-raw-to-rgb"
    mirrors = [ ]

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.image_list = os.listdir(self.root + "/train/canon")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.image_list[index]


class ZurichRawToRgbDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, burst_size: int, patch_size: int, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.burst_size = burst_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        download_and_extract_archive(ZurichRaw2RgbDataset.url, self.data_dir.as_posix())
        

    def setup(self, stage=None):        
        # Split the training dataset into train and validation
        zrr_full = ZurichRaw2RgbDataset((self.data_dir / ZurichRaw2RgbDataset.dirname).as_posix())
        full_len = len(zrr_full)
        
        train_len = int(full_len * 0.75)
        val_len = int(full_len * 0.15)
        test_len = full_len - val_len

        self.zrr_train, self.zrr_val, self.zrr_test = random_split(zrr_full, [train_len, val_len, test_len])

        
        train_zurich_raw2rgb = ZurichRaw2RgbDataset(root=(self.data_dir / ZurichRaw2RgbDataset.dirname).as_posix())
        self.train_data = SyntheticBurst(
            train_zurich_raw2rgb,
            burst_size=self.burst_size,
            crop_sz=self.patch_size,
        )
        self.valid_data = SyntheticBurstVal(root=self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
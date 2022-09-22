from pathlib import Path
from typing import Optional, List
from typing_extensions import Literal
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets.synthetic_burst.train_dataset import TrainDataset
from datasets.synthetic_burst.test_dataset import TestDataset
from datasets.synthetic_burst.val_dataset import ValDataset
from data_modules.utils import prepare_data


class SyntheticBurstDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        data_dir: Path,
        patch_size: int,
        batch_size: int,
        burst_size: int = 8,
        crop_size: int = 384,
        num_workers: int = 0,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.burst_size = burst_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers

    def prepare_data(self):
        prepare_data(
            self.data_dir,
            TestDataset.filename,
            TestDataset.dirname,
            TestDataset.url,
            TestDataset.mirrors,
        )
        prepare_data(
            self.data_dir,
            ValDataset.filename,
            ValDataset.dirname,
            ValDataset.url,
            ValDataset.mirrors,
        )

    def setup(
        self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None
    ) -> None:
        self.train_data = TrainDataset(
            self.dataset, burst_size=self.burst_size, crop_sz=self.crop_size
        )
        self.val_data = ValDataset(self.data_dir / ValDataset.dirname)
        self.test_data = TestDataset(self.data_dir / TestDataset.dirname)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from data_modules.utils import prepare_data
from datasets.synthetic_burst.train_dataset import TrainDataset
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset


class ZurichRaw2RgbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        prepare_data(
            self.data_dir,
            ZurichRaw2RgbDataset.filename,
            ZurichRaw2RgbDataset.dirname,
            ZurichRaw2RgbDataset.url,
            ZurichRaw2RgbDataset.mirrors,
        )

    def setup(self, stage=None):
        self.train_data = ZurichRaw2RgbDataset(
            self.data_dir / ZurichRaw2RgbDataset.dirname
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

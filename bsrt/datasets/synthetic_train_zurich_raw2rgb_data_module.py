from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Generic, TypeVar
from typing_extensions import ClassVar, Type
from datasets.synthetic_burst.train_dataset import TrainDataProcessor
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset
from datasets.utilities.image_folder_data import ImageFolderData
from datasets.utilities.downloadable import Downloadable
import pytorch_lightning as pl
import torchvision
from torch import Tensor
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data.sampler import SequentialSampler

ZuricRaw2RgbData = ImageFolderData[np.uint8]


@dataclass
class SyntheticTrainZurichRaw2RgbDatasetDataModule(pl.LightningDataModule):
    """DataModule for the "Zurich RAW to RGB mapping" dataset."""

    burst_size: int
    crop_size: int

    data_dir: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    prefetch_factor: int
    dataset: ZurichRaw2RgbDataset = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        super().__init__()

        self.dataset = ZurichRaw2RgbDataset(
            data_dir=self.data_dir,
            transform=TrainDataProcessor(burst_size=self.burst_size, crop_sz=self.crop_size),  # type: ignore
        )

    def prepare_data(self) -> None:
        self.dataset.download()

    def train_dataloader(self) -> DataLoader[ZuricRaw2RgbData]:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
        )

from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
from datasets.synthetic_burst.train_dataset import TrainDataProcessor
from datasets.synthetic_burst.val_dataset import ValDataset
from datasets.utilities.image_folder_data import ImageFolderData
from datasets.zurich_raw2rgb_dataset import ZurichRaw2RgbDataset
from torch.utils.data import DataLoader

ZuricRaw2RgbData = ImageFolderData[np.uint8]


@dataclass
class SyntheticTrainValZurichRaw2RgbDataModule(pl.LightningDataModule):
    """DataModule for the "Zurich RAW to RGB mapping" dataset.

    Args:
        burst_size (int): The number of images in each burst.
        crop_size (int): The number of pixels to crop the images to.
        data_dir (str): The directory to download the dataset to.
        batch_size (int): The number of bursts in each batch.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into CUDA pinned memory before returning them.
        persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed.
        drop_last (bool): If ``True``, the data loader will drop the last incomplete batch.
        timeout (float): If positive, the timeout value for collecting a batch from workers.
            Should always be non-negative.
        prefetch_factor (int): Number of samples loaded in advance by each worker.
    """

    burst_size: int
    crop_size: int
    data_dir: str
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    timeout: float = 0.0
    prefetch_factor: int = 2
    train_dataset: ZurichRaw2RgbDataset = field(init=False)
    val_dataset: ValDataset = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        super().__init__()

    def prepare_data(self) -> None:
        # Download the dataset if not present
        ZurichRaw2RgbDataset(data_dir=self.data_dir).download()
        ValDataset(data_dir=self.data_dir).download()

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ZurichRaw2RgbDataset(
            data_dir=self.data_dir,
            transform=TrainDataProcessor(burst_size=self.burst_size, crop_sz=self.crop_size),  # type: ignore
        )
        self.val_dataset = ValDataset(
            data_dir=self.data_dir, burst_size=self.burst_size
        )

    def train_dataloader(self) -> DataLoader[ZuricRaw2RgbData]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
        )

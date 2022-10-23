import os
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from typing_extensions import Literal

from .synthetic_burst.train_dataset import TrainDataProcessor
from .zurich_raw2rgb_dataset import ZurichRaw2RgbDataset


@dataclass
class SyntheticZurichRaw2RgbDataModule(pl.LightningDataModule):
    """DataModule for the "Zurich RAW to RGB mapping" dataset.

    Args:
        burst_size (int): The number of images in each burst.
        crop_size (int): The number of pixels to crop the images to.
        data_dir (str): The directory to download the dataset to.
        batch_size (int): The number of bursts in each batch.
        precision (Literal["bf16", 16, 32]): The precision to use for the data.
        split_ratio (float): The ratio of the dataset to use for training.
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
    precision: Literal["bf16", 16, 32]
    split_ratio: float = 0.8
    num_workers: int = -1
    pin_memory: bool = False
    persistent_workers: bool = False
    drop_last: bool = False
    timeout: float = 0.0
    prefetch_factor: int = 2
    train_dataset: Subset[Tensor] = field(init=False)
    val_dataset: Subset[Tensor] = field(init=False)

    def __post_init__(
        self,
    ) -> None:
        # NOTE: We initialize and assign everything in __post_init__ instead of
        # in prepare_data and setup because it allows us to share the same
        # dataset across processes.
        super().__init__()
        if self.num_workers == -1:
            cpu_count = os.cpu_count()
            assert (
                cpu_count is not None
            ), "Could not determine the number of CPUs and num_workers was not set."
            self.num_workers = cpu_count

        # Download the dataset if not present
        ZurichRaw2RgbDataset(data_dir=self.data_dir).download()
        precision_map = {"bf16": torch.bfloat16, 16: torch.float16, 32: torch.float32}
        dtype = precision_map[self.precision]
        dataset = ZurichRaw2RgbDataset(
            data_dir=self.data_dir,
            transform=TrainDataProcessor(burst_size=self.burst_size, crop_sz=self.crop_size, dtype=dtype),  # type: ignore
        )

        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(
            dataset, [self.split_ratio, 1 - self.split_ratio]
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
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

    def val_dataloader(self) -> DataLoader[Tensor]:
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

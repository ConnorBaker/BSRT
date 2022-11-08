import os
from dataclasses import dataclass, field
from typing import Callable

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, ToTensor
from typing_extensions import Literal

from bsrt.datasets.synthetic_burst.train_dataset import TrainData, TrainDataProcessor


@dataclass(eq=False)
class SyntheticDataModule(pl.LightningDataModule):
    """Synthetic data module for training and validation.

    Args:
        burst_size (int): The number of images in each burst.
        crop_size (int): The number of pixels to crop the images to.
        batch_size (int): The number of bursts in each batch.
        precision (Literal["bf16", 16, 32]): The precision to use for the data.
        dataset (VisionDataset): The dataset to use.
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
    batch_size: int
    precision: Literal["bf16", 16, 32]
    dataset: VisionDataset
    transform: Callable[[TrainData], TrainData]
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

        precision_map = {"bf16": torch.bfloat16, 16: torch.float16, 32: torch.float32}
        dtype = precision_map[self.precision]

        if self.dataset.transform is None:
            self.dataset.transform = Compose([])

        self.dataset.transform.transforms.append(
            TrainDataProcessor(burst_size=self.burst_size, crop_sz=self.crop_size, dtype=dtype)
        )

        train_dataset_len = int(len(self.dataset) * self.split_ratio)
        val_dataset_len = len(self.dataset) - train_dataset_len

        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_dataset_len, val_dataset_len]
        )

    def train_dataloader(self) -> DataLoader[Tensor]:
        return DataLoader(
            self.train_dataset.dataset,
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
            self.val_dataset.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
        )

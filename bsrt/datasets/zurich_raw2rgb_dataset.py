from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from typing_extensions import ClassVar
from datasets.utilities.image_folder_data import ImageFolderData
from datasets.utilities.downloadable import Downloadable
import pytorch_lightning as pl
import torchvision
from torch import Tensor
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
import numpy as np

ZuricRaw2RgbData = ImageFolderData[np.uint8]


@dataclass(init=False, eq=False)
class ZurichRaw2RgbDataset(pl.LightningDataModule, Downloadable):
    """Canon RGB images from the "Zurich RAW to RGB mapping" dataset. You can download the full
    dataset (22 GB) from http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset. Alternatively, you can only download the
    Canon RGB images (5.5 GB) from https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/zurich-raw-to-rgb.zip"
    filename: ClassVar[str] = "zurich-raw-to-rgb.zip"
    dirname: ClassVar[str] = "zurich-raw-to-rgb"
    mirrors: ClassVar[list[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/zurich-raw-to-rgb.zip"
    ]

    data_dir: Path
    batch_size: int = 32
    transform: Callable[[Tensor], Tensor]
    dataset_dir: Path = field(init=False)
    dataset: VisionDataset = field(init=False)

    @dataclass
    class _VisionDataset(VisionDataset):
        path_strs: list[str]
        transform: Callable[[Tensor], Tensor]

        def __getitem__(self, index: int) -> Tensor:
            image_file = torchvision.io.read_file(self.path_strs[index])
            # TODO: Use nvjpg to decode the images more quickly when on CUDA
            image_jpg = torchvision.io.decode_jpeg(image_file)
            transformed = self.transform(image_jpg)
            return transformed

        def __len__(self) -> int:
            return len(self.path_strs)

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        transform: Callable[[Tensor], Tensor] = nn.Identity(),
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_dir: Path = self.data_dir / self.dirname
        self.transform = transform

    def prepare_data(self) -> None:
        self.download()

    def setup(self, stage: None = None) -> None:
        path_strs = [path.as_posix() for path in self.dataset_dir.rglob("*.jpg")]
        self.dataset = self._VisionDataset(path_strs, self.transform)

    def train_dataloader(self) -> DataLoader[ZuricRaw2RgbData]:
        return DataLoader(self.dataset, batch_size=self.batch_size)

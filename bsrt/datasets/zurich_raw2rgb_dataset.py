from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing_extensions import ClassVar
from datasets.utilities.image_folder_data import ImageFolderData
from datasets.utilities.downloadable import Downloadable
from datasets.utilities.provides import ProvidesDataset, ProvidesDatasetPipeline, ProvidesDatasource
from ray.data.datasource import ImageFolderDatasource
from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data import read_datasource
import numpy as np

ZuricRaw2RgbData = ImageFolderData[np.uint8]


@dataclass
class ZurichRaw2RgbDataset(Downloadable, ProvidesDatasource, ProvidesDataset, ProvidesDatasetPipeline):
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

    def provide_dataset_pipeline(
        self, blocks_per_window: Optional[int] = 100
    ) -> DatasetPipeline[ZuricRaw2RgbData]:
        return self.provide_datasource().window(blocks_per_window=blocks_per_window)

    def provide_dataset(self) -> Dataset[ZuricRaw2RgbData]:
        return self.provide_datasource()

    def provide_datasource(self) -> Dataset[ZuricRaw2RgbData]:
        return read_datasource(
            ImageFolderDatasource(),
            root=self.data_dir.as_posix(),
            size=(448, 448),
            mode="RGB",
        ).lazy()

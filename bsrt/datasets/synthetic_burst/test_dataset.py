from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets.utilities.downloadable import Downloadable
from datasets.utilities.image_folder_data import ImageFolderData
from datasets.utilities.provides import (
    ProvidesDataset,
    ProvidesDatasetPipeline,
    ProvidesDatasource,
)
from ray.data import read_datasource
from ray.data.aggregate import AggregateFn
from ray.data.block import DataBatch
from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.datasource import ImageFolderDatasource
from torch import Tensor
from typing_extensions import ClassVar


@dataclass
class TestData:
    """
    burst: LR RAW burst, a torch tensor of shape
        The 4 channels correspond to 'R', 'G', 'G', and 'B' values in the RGGB bayer mosaick.
    meta_info: Meta information about the burst
    """

    burst: Tensor
    meta_info: Dict[str, Any]


@dataclass
class TestDataset(
    Downloadable, ProvidesDatasource, ProvidesDataset, ProvidesDatasetPipeline
):
    """Synthetic burst test set. The test burst have been generated using the same synthetic pipeline as
    employed in SyntheticBurst dataset.
    """

    url: ClassVar[str] = "https://data.vision.ee.ethz.ch/bhatg/synburst_test_2022.zip"
    filename: ClassVar[str] = "synburst_test_2022.zip"
    dirname: ClassVar[str] = "synburst_test_2022"
    mirrors: ClassVar[List[str]] = [
        "https://storage.googleapis.com/bsrt-supplemental/synburst_test_2022.zip"
    ]

    data_dir: Path
    burst_size: ClassVar[int] = 14

    @staticmethod
    def _merge(a1: TestData, a2: TestData) -> TestData:
        burst = torch.vstack([a1.burst, a2.burst])
        return TestData(burst=burst, meta_info=a1.meta_info)

    @staticmethod
    def _accumulate_block(acc: TestData, x: pd.DataFrame) -> TestData:
        np_frames = map(lambda image: image.astype(np.float32), x["image"].to_numpy())
        torch_frames = map(
            lambda np_frame: torch.from_numpy(np_frame).permute(2, 0, 1), np_frames
        )
        normalized_torch_frames = map(
            lambda torch_frame: torch_frame.float() / (2**14), torch_frames
        )
        burst = torch.stack(list(normalized_torch_frames))
        return TestData(burst=burst, meta_info=acc.meta_info)

    _aggregate_fn: ClassVar[AggregateFn] = AggregateFn(
        init=lambda burst_name: TestData(
            burst=torch.empty(0), meta_info={"burst_name": burst_name}
        ),
        accumulate_block=_accumulate_block,  # type: ignore
        merge=_merge,
        name="test_data",
    )

    @staticmethod
    def _unnest_dataframe(df: DataBatch) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame)
        return pd.DataFrame(df["test_data"].tolist())

    def provide_dataset_pipeline(
        self, blocks_per_window: int = 100
    ) -> DatasetPipeline[TestData]:
        return self.provide_dataset().window(blocks_per_window=blocks_per_window)

    def provide_dataset(self) -> Dataset[TestData]:
        return (
            self.provide_datasource()
            .groupby("label")
            .aggregate(self._aggregate_fn)
            .map_batches(TestDataset._unnest_dataframe, batch_format="pandas")
        )

    def provide_datasource(self) -> Dataset[ImageFolderData[np.uint8]]:
        return read_datasource(
            ImageFolderDatasource(),
            root=self.data_dir.as_posix(),
            size=(128, 128),
            mode="RGB",
        ).lazy()

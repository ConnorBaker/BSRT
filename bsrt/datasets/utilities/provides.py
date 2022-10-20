from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ray.data.dataset import Dataset
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.datasource import Datasource

_T = TypeVar("_T")


class ProvidesDatasetPipeline(ABC, Generic[_T]):
    @abstractmethod
    def provide_dataset_pipeline(
        self, blocks_per_window: int = 100
    ) -> DatasetPipeline[_T]:
        ...


class ProvidesDataset(ABC, Generic[_T]):
    @abstractmethod
    def provide_dataset(self) -> Dataset[_T]:
        ...


class ProvidesDatasource(ABC, Generic[_T]):
    @abstractmethod
    def provide_datasource(self) -> Datasource[_T]:
        ...

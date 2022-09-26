from abc import abstractmethod
from typing import Generic, TypeVar
from abc import ABC
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.dataset import Dataset
from ray.data.datasource import Datasource

_T = TypeVar('_T')

class TransformsDatasetPipeline(ABC, Generic[_T]):
    @abstractmethod
    def transform_dataset_pipeline(self, dataset_pipeline: DatasetPipeline[_T]) -> DatasetPipeline[_T]: ...

class TransformsDataset(ABC, Generic[_T]):
    @abstractmethod
    def transform_dataset(self, dataset: Dataset[_T]) -> Dataset[_T]: ...

class TransformsDatasource(ABC, Generic[_T]):
    @abstractmethod
    def transform_datasource(self, datasource: Datasource[_T]) -> Datasource[_T]: ...
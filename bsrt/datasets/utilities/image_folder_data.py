from typing import Generic, TypeVar
from typing_extensions import TypedDict, TypedDict
import numpy.typing as npt
import numpy as np

_T = TypeVar('_T', bound=np.number)

class ImageFolderData(TypedDict, Generic[_T]):
    image: npt.NDArray[_T]
    label: str

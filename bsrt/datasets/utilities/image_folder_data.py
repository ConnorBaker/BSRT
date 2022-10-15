from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import TypedDict

_T = TypeVar("_T", bound=np.number)


class ImageFolderData(TypedDict, Generic[_T]):
    image: npt.NDArray[_T]
    label: str

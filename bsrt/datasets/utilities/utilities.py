from typing import TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import torch

_T = TypeVar("_T", bound=np.floating)


@overload
def pack_raw_image(im_raw: npt.NDArray[_T]) -> npt.NDArray[_T]:
    ...


@overload
def pack_raw_image(im_raw: torch.Tensor) -> torch.Tensor:
    ...


def pack_raw_image(
    im_raw: Union[npt.NDArray[_T], torch.Tensor]
) -> Union[npt.NDArray[_T], torch.Tensor]:
    im_out: Union[npt.NDArray[_T], torch.Tensor]
    if isinstance(im_raw, np.ndarray):
        im_out = np.zeros_like(im_raw, shape=(4, im_raw.shape[0] // 2, im_raw.shape[1] // 2))
        for i in range(4):
            im_out[i, :, :] = im_raw[(i // 2) :: 2, i::2]
        return im_out
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
        for i in range(4):
            im_out[i, :, :] = im_raw[(i // 2) :: 2, i::2]
        return im_out
    else:
        raise Exception


@overload
def flatten_raw_image(im_raw_4ch: npt.NDArray[_T]) -> npt.NDArray[_T]:
    ...


@overload
def flatten_raw_image(im_raw_4ch: torch.Tensor) -> torch.Tensor:
    ...


def flatten_raw_image(
    im_raw_4ch: Union[npt.NDArray[_T], torch.Tensor]
) -> Union[npt.NDArray[_T], torch.Tensor]:
    im_out: Union[npt.NDArray[_T], torch.Tensor]
    if isinstance(im_raw_4ch, np.ndarray):
        im_out = np.zeros_like(
            im_raw_4ch, shape=(im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2)
        )
        for i in range(4):
            im_out[(i // 2) :: 2, i::2] = im_raw_4ch[i, :, :]
        return im_out

    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros(
            (im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype
        )
        for i in range(4):
            im_out[(i // 2) :: 2, i::2] = im_raw_4ch[i, :, :]
        return im_out
    else:
        raise Exception

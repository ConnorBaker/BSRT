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
    elif isinstance(im_raw, torch.Tensor):
        im_out = torch.zeros((4, im_raw.shape[0] // 2, im_raw.shape[1] // 2), dtype=im_raw.dtype)
    else:
        raise Exception

    # Manually unroll the assignment loop because it's faster.
    # Notice that we're effectively counting in binary on the right hand side.
    im_out[0, :, :] = im_raw[0::2, 0::2]  # type: ignore
    im_out[1, :, :] = im_raw[0::2, 1::2]  # type: ignore
    im_out[2, :, :] = im_raw[1::2, 0::2]  # type: ignore
    im_out[3, :, :] = im_raw[1::2, 1::2]  # type: ignore

    return im_out


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

    elif isinstance(im_raw_4ch, torch.Tensor):
        im_out = torch.zeros(
            (im_raw_4ch.shape[1] * 2, im_raw_4ch.shape[2] * 2), dtype=im_raw_4ch.dtype
        )
    else:
        raise Exception

    # Manually unroll the assignment loop because it's faster.
    # Notice that we're effectively counting in binary on the left hand side.
    im_out[0::2, 0::2] = im_raw_4ch[0, :, :]  # type: ignore
    im_out[0::2, 1::2] = im_raw_4ch[1, :, :]  # type: ignore
    im_out[1::2, 0::2] = im_raw_4ch[2, :, :]  # type: ignore
    im_out[1::2, 1::2] = im_raw_4ch[3, :, :]  # type: ignore

    return im_out

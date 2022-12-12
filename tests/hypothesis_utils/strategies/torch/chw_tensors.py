from typing import NewType

import torch
from hypothesis import strategies as st

from tests.hypothesis_utils.strategies.torch.devices_and_dtypes import devices_and_dtypes

CHWShape = NewType("CHWShape", tuple[int, int, int])


@st.composite
def chw_shapes(
    draw: st.DrawFn,
    *,
    min_channels: int = 1,
    max_channels: None | int = None,
    min_height: int = 1,
    max_height: None | int = None,
    min_width: int = 1,
    max_width: None | int = None,
) -> CHWShape:
    """
    Returns a strategy for generating shapes for CHW tensors. Useful for testing image processing.

    Args:
        min_channels: Minimum number of channels.
        max_channels: Maximum number of channels.
        min_height: Minimum height.
        max_height: Maximum height.
        min_width: Minimum width.
        max_width: Maximum width.

    Returns:
        A strategy for generating shapes for CHW tensors.
    """
    channels = draw(st.integers(min_channels, max_channels))
    height = draw(st.integers(min_height, max_height))
    width = draw(st.integers(min_width, max_width))
    return CHWShape((channels, height, width))


@st.composite
def chw_tensors(
    draw: st.DrawFn,
    *,
    shape: None | CHWShape | st.SearchStrategy[CHWShape] = None,
    dtype: None | torch.dtype | st.SearchStrategy[torch.dtype] = None,
    device: None | str | torch.device | st.SearchStrategy[torch.device] = None,
) -> torch.Tensor:
    """
    Returns a tensor with the given dtype and shape. If either is None, it will be sampled.

    Args:
        shape: The shape of the tensor. If None, a random shape will be generated.
        dtype: The dtype of the tensor. If None, it will be sampled.
        device: The device of the tensor. If None, it will be sampled.

    Returns:
        A tensor with the given dtype and shape on the given device.
    """
    if shape is None:
        shape = draw(chw_shapes())
    elif isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    assert len(shape) == 3, "Shape must be 3-dimensional"
    assert all(s > 0 for s in shape), "Shape must have positive components"

    device_and_dtype = draw(devices_and_dtypes(device=device, dtype=dtype))
    empty_like = torch.empty(size=shape, **device_and_dtype)
    if empty_like.dtype.is_floating_point:
        return empty_like.uniform_()
    elif not empty_like.dtype.is_complex:
        return empty_like.random_()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

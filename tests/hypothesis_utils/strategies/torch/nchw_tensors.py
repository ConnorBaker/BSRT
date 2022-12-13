from typing import NewType

import torch
from hypothesis import strategies as st

from tests.hypothesis_utils.strategies.torch.chw_tensors import chw_shapes
from tests.hypothesis_utils.strategies.torch.devices_and_dtypes import devices_and_dtypes
from tests.hypothesis_utils.strategies.torch.memory_formats import torch_memory_formats

NCHWShape = NewType("NCHWShape", tuple[int, int, int, int])


@st.composite
def nchw_shapes(
    draw: st.DrawFn,
    *,
    min_batch: int = 1,
    max_batch: None | int = None,
    min_channels: int = 1,
    max_channels: None | int = None,
    min_height: int = 1,
    max_height: None | int = None,
    min_width: int = 1,
    max_width: None | int = None,
) -> NCHWShape:
    """
    Returns a strategy for generating shapes for NCHW tensors. Useful for testing batches of
    images.

    Args:
        min_batch: Minimum batch size.
        max_batch: Maximum batch size.
        min_channels: Minimum number of channels.
        max_channels: Maximum number of channels.
        min_height: Minimum height.
        max_height: Maximum height.
        min_width: Minimum width.
        max_width: Maximum width.

    Returns:
        A strategy for generating shapes for NCHW tensors.
    """
    batch = draw(st.integers(min_batch, max_batch))
    chw_shape = draw(
        chw_shapes(
            min_channels=min_channels,
            max_channels=max_channels,
            min_height=min_height,
            max_height=max_height,
            min_width=min_width,
            max_width=max_width,
        )
    )
    return NCHWShape((batch, *chw_shape))


@st.composite
def nchw_tensors(
    draw: st.DrawFn,
    *,
    shape: None | NCHWShape | st.SearchStrategy[NCHWShape] = None,
    dtype: None | torch.dtype | st.SearchStrategy[torch.dtype] = None,
    device: None | str | torch.device | st.SearchStrategy[torch.device] = None,
    memory_format: None | torch.memory_format | st.SearchStrategy[torch.memory_format] = None,
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
        shape = draw(nchw_shapes())
    elif isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    assert len(shape) == 4, "Shape must be 4-dimensional"
    assert all(s > 0 for s in shape), "Shape must have positive components"

    device_and_dtype = draw(devices_and_dtypes(device=device, dtype=dtype))

    if memory_format is None:
        memory_format = draw(torch_memory_formats)
    elif isinstance(memory_format, st.SearchStrategy):
        memory_format = draw(memory_format)

    empty_like = torch.empty(size=shape, **device_and_dtype, memory_format=memory_format)
    if empty_like.dtype.is_floating_point:
        return empty_like.uniform_()
    elif not empty_like.dtype.is_complex:
        return empty_like.random_()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@st.composite
def nchw_tensors_with_same_shape_and_device(
    draw: st.DrawFn,
    *,
    shape: None | NCHWShape | st.SearchStrategy[NCHWShape] = None,
    dtype: None | torch.dtype | st.SearchStrategy[torch.dtype] = None,
    device: None | str | torch.device | st.SearchStrategy[torch.device] = None,
    memory_format: None | torch.memory_format | st.SearchStrategy[torch.memory_format] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns two tensors with the same shape on the same device.

    Args:
        shape: The shape of the tensor. If None, it will be sampled.
        dtype: The dtype of the tensor. If None, it will be sampled.
        device: The device of the tensor. If None, it will be sampled.

    Returns:
        A tuple of two tensors with the same shape and dtype on the same device.
    """

    # There's no easy way to get the memory format of a tensor, so we just draw it here.
    if memory_format is None:
        memory_format = draw(torch_memory_formats)
    elif isinstance(memory_format, st.SearchStrategy):
        memory_format = draw(memory_format)

    t1 = draw(
        nchw_tensors(
            dtype=dtype,
            device=device,
            shape=shape,
            memory_format=memory_format,
        )
    )
    t2 = draw(
        nchw_tensors(
            dtype=t1.dtype,
            device=t1.device,
            shape=NCHWShape(t1.shape),
            memory_format=memory_format,
        )
    )
    return t1, t2

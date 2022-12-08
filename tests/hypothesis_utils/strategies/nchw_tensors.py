import torch
from hypothesis import strategies as st
from typing import NewType
from tests.hypothesis_utils.strategies.chw_tensors import chw_shapes
from tests.hypothesis_utils.strategies.torch_devices import torch_devices
from tests.hypothesis_utils.strategies.torch_dtypes import torch_float_dtypes


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

    assert len(shape) == 4, "Shape must be 3-dimensional"
    assert all(s > 0 for s in shape), "Shape must have positive components"

    if dtype is None:
        dtype = draw(torch_float_dtypes)
    elif isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)

    if device is None:
        device = draw(torch_devices)
    elif isinstance(device, st.SearchStrategy):
        device = draw(device)

    if dtype.is_floating_point:
        return torch.rand(shape, dtype=dtype, device=device)
    elif not dtype.is_complex:
        iinfo = torch.iinfo(dtype)
        return torch.randint(
            low=iinfo.min,
            high=iinfo.max,
            size=shape,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@st.composite
def nchw_tensors_with_same_shape_and_device(
    draw: st.DrawFn,
    *,
    shape: None | NCHWShape | st.SearchStrategy[NCHWShape] = None,
    dtype: None | torch.dtype | st.SearchStrategy[torch.dtype] = None,
    device: None | str | torch.device | st.SearchStrategy[torch.device] = None,
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
    t1 = draw(
        nchw_tensors(
            dtype=dtype,
            device=device,
            shape=shape,
        )
    )
    t2 = draw(
        nchw_tensors(
            dtype=t1.dtype,
            device=t1.device,
            shape=NCHWShape(t1.shape),
        )
    )
    return t1, t2

from typing import TypedDict

import torch
from hypothesis import strategies as st
from torch.backends import mps

from tests.hypothesis_utils.strategies.torch.devices import torch_devices
from tests.hypothesis_utils.strategies.torch.dtypes import torch_real_dtypes


class DeviceAndDType(TypedDict):
    device: torch.device
    dtype: torch.dtype


# Checks for consistency with device and dtype selection
@st.composite
def devices_and_dtypes(
    draw: st.DrawFn,
    *,
    device: None | str | torch.device | st.SearchStrategy[torch.device] = None,
    dtype: None | torch.dtype | st.SearchStrategy[torch.dtype] = None,
) -> DeviceAndDType:
    if device is None:
        device = draw(torch_devices)
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, st.SearchStrategy):
        device = draw(device)

    if dtype is None:
        dtype_strat = torch_real_dtypes
    elif isinstance(dtype, torch.dtype):
        dtype_strat = st.just(dtype)
    elif isinstance(dtype, st.SearchStrategy):
        dtype_strat = dtype

    # Filter out unsupported dtypes
    if device.type == "mps":
        # MPS doesn't support bfloat16 or float64
        dtype_strat = dtype_strat.filter(lambda t: t not in [torch.bfloat16, torch.float64])

    if device.type == "cpu" and mps.is_available():
        # Apple M1 doesn't float16
        dtype_strat = dtype_strat.filter(lambda t: t != torch.float16)

    dtype = draw(dtype_strat)

    return DeviceAndDType(device=device, dtype=dtype)

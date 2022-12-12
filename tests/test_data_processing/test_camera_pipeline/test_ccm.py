import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import RandomSeeder
from torch import Tensor

from bsrt.data_processing.camera_pipeline import apply_ccm, random_ccm
from tests.hypothesis_utils.strategies.torch._3hw_tensors import _3HW_TENSORS
from tests.hypothesis_utils.strategies.torch.devices_and_dtypes import (
    DeviceAndDType,
    devices_and_dtypes,
)
from tests.hypothesis_utils.strategies.torch.dtypes import torch_float_dtypes

# Property-based tests which ensure:
# - The CCM from random_ccm() is 3x3
# - The CCM from random_ccm() is invariant with respect to the dtype
# - The CCM from random_ccm() is invariant with respect to the device
# - The CCM from random_ccm() has rows which sum to 1
# - apply_ccm() is invariant with respect to the image shape
# - apply_ccm() is invariant with respect to the image dtype
# - apply_ccm() is invariant with respect to the image device


@given(
    rs=st.random_module(),
    meta=devices_and_dtypes(dtype=torch_float_dtypes),
)
def test_random_ccm_shape(rs: RandomSeeder, meta: DeviceAndDType) -> None:
    """
    Tests that the CCM from random_ccm() is 3x3.

    Args:
        rs: A random seed
    """
    expected = torch.Size([3, 3])
    actual = random_ccm(**meta).shape
    assert actual == expected


@given(
    rs=st.random_module(),
    meta=devices_and_dtypes(dtype=torch_float_dtypes),
)
def test_random_ccm_dtype(rs: RandomSeeder, meta: DeviceAndDType) -> None:
    """
    Tests that the CCM from random_ccm() is invariant with respect to the dtype.

    Args:
        rs: A random seed
    """
    expected = meta["dtype"]
    actual = random_ccm(**meta).dtype
    assert actual == expected


@given(
    rs=st.random_module(),
    meta=devices_and_dtypes(dtype=torch_float_dtypes),
)
def test_random_ccm_device(rs: RandomSeeder, meta: DeviceAndDType) -> None:
    """
    Tests that the CCM from random_ccm() is invariant with respect to the device.

    Args:
        rs: A random seed
    """
    expected = meta["device"]
    actual = random_ccm(**meta).device
    assert actual == expected


@given(
    rs=st.random_module(),
    meta=devices_and_dtypes(dtype=torch_float_dtypes),
)
def test_random_ccm_rows_sum_to_1(rs: RandomSeeder, meta: DeviceAndDType) -> None:
    """
    Tests that the CCM from random_ccm() has rows which sum to 1.

    Args:
        rs: A random seed
    """
    expected = torch.ones(3, **meta)
    actual = random_ccm(**meta).sum(dim=1)
    match meta["dtype"]:
        case torch.bfloat16:
            rtol = 1e-02
        case torch.float16:
            rtol = 1e-03
        case torch.float32:
            rtol = 1e-04
        case _:
            rtol = 1e-05
    assert torch.allclose(actual, expected, rtol=rtol)


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_shape_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image shape.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A randomdef test_apply_ccm_shape_in
    """
    expected = image.shape
    ccm = random_ccm(dtype=image.dtype, device=image.device)
    actual = apply_ccm(image, ccm).shape
    assert actual == expected


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_dtype_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image dtype.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A random seed
    """
    expected = image.dtype
    ccm = random_ccm(dtype=image.dtype, device=image.device)
    actual = apply_ccm(image, ccm).dtype
    assert actual == expected


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_device_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image device.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A random
    """
    expected = image.device
    ccm = random_ccm(dtype=image.dtype, device=image.device)
    actual = apply_ccm(image, ccm).device
    assert actual == expected

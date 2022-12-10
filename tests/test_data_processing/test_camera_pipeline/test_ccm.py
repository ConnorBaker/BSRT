import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import RandomSeeder
from torch import Tensor

from bsrt.data_processing.camera_pipeline import apply_ccm, random_ccm
from tests.hypothesis_utils.strategies.torch._3hw_tensors import _3HW_TENSORS

# Property-based tests which ensure:
# - The CCM from random_ccm() is 3x3
# - The CCM from random_ccm() is float32
# - The CCM from random_ccm() has rows which sum to 1
# - apply_ccm() is invariant with respect to the image shape
# - apply_ccm() is invariant with respect to the image dtype
# - apply_ccm() is invariant with respect to the image device


@given(rs=st.random_module())
def test_random_ccm_shape(rs: RandomSeeder) -> None:
    """
    Tests that the CCM from random_ccm() is 3x3.

    Args:
        rs: A random seed
    """
    assert random_ccm().shape == torch.Size([3, 3])


@given(rs=st.random_module())
def test_random_ccm_dtype(rs: RandomSeeder) -> None:
    """
    Tests that the CCM from random_ccm() is float32.

    Args:
        rs: A random seed
    """
    assert random_ccm().dtype == torch.float32


@given(rs=st.random_module())
def test_random_ccm_rows_sum_to_1(rs: RandomSeeder) -> None:
    """
    Tests that the CCM from random_ccm() has rows which sum to 1.

    Args:
        rs: A random seed
    """
    assert torch.allclose(random_ccm().sum(dim=1), torch.ones(3))


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_shape_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image shape.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A randomdef test_apply_ccm_shape_in
    """
    ccm = random_ccm()
    assert apply_ccm(image, ccm).shape == image.shape


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_dtype_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image dtype.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A random seed
    """
    ccm = random_ccm()
    assert apply_ccm(image, ccm).dtype == image.dtype


@given(image=_3HW_TENSORS(), rs=st.random_module())
def test_apply_ccm_device_invariance(image: Tensor, rs: RandomSeeder) -> None:
    """
    Tests that apply_ccm() is invariant with respect to the image device.

    Args:
        image: A 3HW tensor of floating dtype
        rs: A random
    """
    ccm = random_ccm()
    assert apply_ccm(image, ccm).device == image.device

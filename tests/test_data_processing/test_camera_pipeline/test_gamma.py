import torch
from hypothesis import given
from torch import Tensor

from bsrt.data_processing.camera_pipeline import gamma_compression, gamma_expansion
from tests.hypothesis_utils.strategies.torch._3hw_tensors import _3HW_TENSORS

# Property-based tests which ensure:
# - gamma_expansion is invariant with respect to shape
# - gamma_compression is invariant with respect to shape
# - gamma_expansion is invariant with respect to dtype
# - gamma_compression is invariant with respect to dtype
# - gamma_expansion is invariant with respect to device
# - gamma_compression is invariant with respect to device
# - gamma_expansion is the inverse of gamma_compression (roughly)
# - gamma_compression is the inverse of gamma_expansion (roughly)


@given(image=_3HW_TENSORS())
def test_gamma_expansion_shape_invariance(image: Tensor) -> None:
    """
    Tests that gamma_expansion is invariant with respect to shape.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.shape
    actual = gamma_expansion(image).shape
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_compression_shape_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to shape.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.shape
    actual = gamma_compression(image).shape
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_expansion_dtype_invariance(image: Tensor) -> None:
    """
    Tests that gamma_expansion is invariant with respect to dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.dtype
    actual = gamma_expansion(image).dtype
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_compression_dtype_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.dtype
    actual = gamma_compression(image).dtype
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_expansion_device_invariance(image: Tensor) -> None:
    """
    Tests that gamma_expansion is invariant with respect to device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.device
    actual = gamma_expansion(image).device
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_compression_device_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image.device
    actual = gamma_compression(image).device
    assert actual == expected


@given(image=_3HW_TENSORS())
def test_gamma_expansion_is_inverse_of_gamma_compression(image: Tensor) -> None:
    """
    Tests that gamma_expansion is the inverse of gamma_compression (roughly).

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image
    actual = gamma_expansion(gamma_compression(image))
    assert torch.allclose(expected, actual, rtol=1e-2, atol=1e-3)


@given(image=_3HW_TENSORS())
def test_gamma_compression_is_inverse_of_gamma_expansion(image: Tensor) -> None:
    """
    Tests that gamma_compression is the inverse of gamma_expansion (roughly).

    Args:
        image: A 3HW tensor of floating dtype
    """
    expected = image
    actual = gamma_compression(gamma_expansion(image))
    assert torch.allclose(expected, actual, rtol=1e-2, atol=1e-3)

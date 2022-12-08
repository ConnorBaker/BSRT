import torch
from torch import Tensor
from hypothesis import given
from tests.hypothesis_utils.strategies._3hw_tensors import _3HW_TENSORS
from bsrt.data_processing.camera_pipeline import gamma_compression, gamma_expansion


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
    assert image.shape == gamma_expansion(image).shape


@given(image=_3HW_TENSORS())
def test_gamma_compression_shape_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to shape.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert image.shape == gamma_compression(image).shape


@given(image=_3HW_TENSORS())
def test_gamma_expansion_dtype_invariance(image: Tensor) -> None:
    """
    Tests that gamma_expansion is invariant with respect to dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert image.dtype == gamma_expansion(image).dtype


@given(image=_3HW_TENSORS())
def test_gamma_compression_dtype_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert image.dtype == gamma_compression(image).dtype


@given(image=_3HW_TENSORS())
def test_gamma_expansion_device_invariance(image: Tensor) -> None:
    """
    Tests that gamma_expansion is invariant with respect to device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert image.device == gamma_expansion(image).device


@given(image=_3HW_TENSORS())
def test_gamma_compression_device_invariance(image: Tensor) -> None:
    """
    Tests that gamma_compression is invariant with respect to device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert image.device == gamma_compression(image).device


@given(image=_3HW_TENSORS())
def test_gamma_expansion_is_inverse_of_gamma_compression(image: Tensor) -> None:
    """
    Tests that gamma_expansion is the inverse of gamma_compression (roughly).

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert torch.allclose(image, gamma_expansion(gamma_compression(image)), rtol=1e-2, atol=1e-3)


@given(image=_3HW_TENSORS())
def test_gamma_compression_is_inverse_of_gamma_expansion(image: Tensor) -> None:
    """
    Tests that gamma_compression is the inverse of gamma_expansion (roughly).

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert torch.allclose(image, gamma_compression(gamma_expansion(image)), rtol=1e-2, atol=1e-3)

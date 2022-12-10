from hypothesis import given
from torch import Tensor
from torch.nn import functional as F

from bsrt.data_processing.camera_pipeline import apply_smoothstep, invert_smoothstep
from tests.hypothesis_utils.strategies._3hw_tensors import _3HW_TENSORS

# Property-based tests which ensure:
# - apply_smoothstep is invariant with respect to shape
# - invert_smoothstep is invariant with respect to shape
# - apply_smoothstep is invariant with respect to dtype
# - invert_smoothstep is invariant with respect to dtype
# - apply_smoothstep is invariant with respect to device
# - invert_smoothstep is invariant with respect to device
# - invert_smoothstep is the inverse of apply_smoothstep (roughly)


@given(image=_3HW_TENSORS())
def test_apply_smoothstep_shape_invariance(image: Tensor) -> None:
    """
    Tests that apply_smoothstep() is invariant with respect to the image shape.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert apply_smoothstep(image).shape == image.shape


@given(image=_3HW_TENSORS())
def test_invert_smoothstep_shape_invariance(image: Tensor) -> None:
    """
    Tests that invert_smoothstep() is invariant with respect to the image shape.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert invert_smoothstep(image).shape == image.shape


@given(image=_3HW_TENSORS())
def test_apply_smoothstep_dtype_invariance(image: Tensor) -> None:
    """
    Tests that apply_smoothstep() is invariant with respect to the image dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert apply_smoothstep(image).dtype == image.dtype


@given(image=_3HW_TENSORS())
def test_invert_smoothstep_dtype_invariance(image: Tensor) -> None:
    """
    Tests that invert_smoothstep() is invariant with respect to the image dtype.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert invert_smoothstep(image).dtype == image.dtype


@given(image=_3HW_TENSORS())
def test_apply_smoothstep_device_invariance(image: Tensor) -> None:
    """
    Tests that apply_smoothstep() is invariant with respect to the image device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert apply_smoothstep(image).device == image.device


@given(image=_3HW_TENSORS())
def test_invert_smoothstep_device_invariance(image: Tensor) -> None:
    """
    Tests that invert_smoothstep() is invariant with respect to the image device.

    Args:
        image: A 3HW tensor of floating dtype
    """
    assert invert_smoothstep(image).device == image.device


@given(image=_3HW_TENSORS())
def test_invert_smoothstep_is_inverse_of_apply_smoothstep(image: Tensor) -> None:
    """
    Tests that invert_smoothstep() is the inverse of apply_smoothstep().

    Args:
        image: A 3HW tensor of floating dtype
    """
    smoothstep_inverted = invert_smoothstep(image)
    smoothstep_applied = apply_smoothstep(smoothstep_inverted)

    # The smoothstep function is not invertible, so we can only test that the result is close to
    # the original image.
    image_to_inverted_smoothstep_mse = F.mse_loss(image, smoothstep_inverted)
    image_to_smoothstep_applied_mse = F.mse_loss(image, smoothstep_applied)
    assert image_to_smoothstep_applied_mse < image_to_inverted_smoothstep_mse
    assert image_to_smoothstep_applied_mse <= 1e-4

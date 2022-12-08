from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bsrt.data_processing.synthetic_burst_generator import (
    get_tmat,
    pure_python_get_tmat,
    pure_python_get_tmat_fast1,
    pure_python_get_tmat_fast2,
    pure_python_get_tmat_fast3,
)


def given_get_tmat_args(f):
    return given(
        image_shape=st.tuples(
            st.integers(min_value=32, max_value=1000),
            st.integers(min_value=32, max_value=1000),
        ),
        translation=st.tuples(
            st.floats(min_value=-8, max_value=8),
            st.floats(min_value=-8, max_value=8),
        ),
        theta=st.floats(min_value=-180.0, max_value=180.0),
        shear_values=st.tuples(
            st.floats(min_value=-1.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
        ),
        scale_factors=st.tuples(
            st.floats(min_value=0.01, max_value=1.0),
            st.floats(min_value=0.01, max_value=1.0),
        ),
    )(f)


@given_get_tmat_args
def test_get_tmat_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert tmat.shape == (2, 3), f"{tmat.shape} != (3, 3)"


@given_get_tmat_args
def test_get_tmat_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
) -> None:
    tmat = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert tmat.dtype == np.float64, f"{tmat.dtype} != np.float64"


@pytest.mark.parametrize(
    "impl",
    [
        pure_python_get_tmat,
        pure_python_get_tmat_fast1,
        pure_python_get_tmat_fast2,
        pure_python_get_tmat_fast3,
    ],
)
@given_get_tmat_args
def test_get_tmat_shape_eq_impl_shape(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64]],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert expected.shape == actual.shape, f"{expected.shape} != {actual.shape}"


@pytest.mark.parametrize(
    "impl",
    [
        pure_python_get_tmat,
        pure_python_get_tmat_fast1,
        pure_python_get_tmat_fast2,
        pure_python_get_tmat_fast3,
    ],
)
@given_get_tmat_args
def test_get_tmat_dtype_eq_impl_dtype(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64]],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert expected.dtype == actual.dtype, f"{expected.dtype} != {actual.dtype}"


@pytest.mark.parametrize(
    "impl",
    [
        pure_python_get_tmat,
        pure_python_get_tmat_fast1,
        pure_python_get_tmat_fast2,
        pure_python_get_tmat_fast3,
    ],
)
@given_get_tmat_args
def test_get_tmat_values_eq_impl_values(
    image_shape: tuple[int, int],
    translation: tuple[float, float],
    theta: float,
    shear_values: tuple[float, float],
    scale_factors: tuple[float, float],
    impl: Callable[..., npt.NDArray[np.float64]],
) -> None:
    expected = get_tmat(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    actual = impl(
        image_shape,
        translation,
        theta,
        shear_values,
        scale_factors,
    )
    assert np.allclose(expected, actual), f"{expected} != {actual}"


# TODO: FAST4 which has just one matrix

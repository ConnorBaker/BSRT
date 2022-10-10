from typing_extensions import overload
from torch import Tensor


@overload
def ignore_boundary(x: None, boundary_ignore: int | None) -> None:
    ...


@overload
def ignore_boundary(x: Tensor, boundary_ignore: int | None) -> Tensor:
    ...


def ignore_boundary(
    x: Tensor | None, boundary_ignore: int | None = None
) -> Tensor | None:
    if boundary_ignore is not None and x is not None:
        x = x[
            ...,
            boundary_ignore:-boundary_ignore,
            boundary_ignore:-boundary_ignore,
        ]

    return x

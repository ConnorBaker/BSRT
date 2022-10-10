from typing import Optional, overload
from torch import Tensor


@overload
def ignore_boundary(x: None, boundary_ignore: Optional[int]) -> None:
    ...


@overload
def ignore_boundary(x: Tensor, boundary_ignore: Optional[int]) -> Tensor:
    ...


def ignore_boundary(
    x: Optional[Tensor], boundary_ignore: Optional[int] = None
) -> Optional[Tensor]:
    if boundary_ignore is not None and x is not None:
        x = x[
            ...,
            boundary_ignore:-boundary_ignore,
            boundary_ignore:-boundary_ignore,
        ]

    return x

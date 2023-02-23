from __future__ import annotations

from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import Callable, cast

from typing_extensions import Literal, Type, TypeVar, get_args, overload

_T = TypeVar("_T", bound="Params")

_SUPPORTED_TYPES = Literal[
    "bool",
    "int",
    "float",
    "str",
    "Optional[bool]",
    "Optional[int]",
    "Optional[float]",
    "Optional[str]",
]


@overload
def str_type_to_type(type_str: Literal["bool"]) -> Callable[..., bool]:
    ...


@overload
def str_type_to_type(type_str: Literal["int"]) -> Callable[..., int]:
    ...


@overload
def str_type_to_type(type_str: Literal["float"]) -> Callable[..., float]:
    ...


@overload
def str_type_to_type(type_str: Literal["str"]) -> Callable[..., str]:
    ...


@overload
def str_type_to_type(type_str: Literal["Optional[bool]"]) -> Callable[..., None | bool]:
    ...


@overload
def str_type_to_type(type_str: Literal["Optional[int]"]) -> Callable[..., None | int]:
    ...


@overload
def str_type_to_type(type_str: Literal["Optional[float]"]) -> Callable[..., None | float]:
    ...


@overload
def str_type_to_type(type_str: Literal["Optional[str]"]) -> Callable[..., None | str]:
    ...


def str_type_to_type(type_str: str) -> Callable[..., None | bool | int | float | str]:
    match type_str:
        case "bool":
            return bool
        case "int":
            return int
        case "float":
            return float
        case "str":
            return str
        case optional if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner = optional.lstrip("Optional[").rstrip("]")
            return lambda y: str_type_to_type(inner)(y) if y is not None else None
        case _:
            raise ValueError(f"Unsupported type: {type_str}")


@dataclass
class Params(ABC):
    @classmethod
    def add_to_argparse(cls, parser: ArgumentParser) -> None:
        arg_group = parser.add_argument_group(cls.__name__)
        for field in fields(cls):
            assert field.type in get_args(_SUPPORTED_TYPES)
            field_type: _SUPPORTED_TYPES = cast(_SUPPORTED_TYPES, field.type)
            arg_group.add_argument(
                f"--{cls.__name__}.{field.name}", type=str_type_to_type(field_type)
            )

    @classmethod
    def from_args(
        cls: Type[_T], args: Namespace  # type: ignore[valid-type]
    ) -> _T:  # type: ignore[valid-type]
        kwargs = {}
        for field in fields(cls):
            assert field.type in get_args(_SUPPORTED_TYPES)
            field_type: _SUPPORTED_TYPES = cast(_SUPPORTED_TYPES, field.type)
            kwargs[field.name] = str_type_to_type(field_type)(
                getattr(args, f"{cls.__name__}.{field.name}")
            )
        instance: _T = cast(Callable[..., _T], cls)(**kwargs)  # type: ignore[valid-type]
        return instance

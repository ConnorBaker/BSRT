from __future__ import annotations

from abc import ABC
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import Type, TypeVar

_T = TypeVar("_T", bound="Params")

STR_TO_TYPE = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}


@dataclass
class Params(ABC):
    @classmethod
    def add_to_argparse(cls, parser: ArgumentParser) -> None:
        arg_group = parser.add_argument_group(cls.__name__)
        for field in fields(cls):
            arg_group.add_argument(
                f"--{cls.__name__}.{field.name}", type=STR_TO_TYPE[field.type], required=True
            )

    @classmethod
    def from_args(cls: Type[_T], args: Namespace) -> _T:
        return cls(
            **{
                field.name: STR_TO_TYPE[field.type](getattr(args, f"{cls.__name__}.{field.name}"))
                for field in fields(cls)
            }
        )

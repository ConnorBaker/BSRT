from dataclasses import dataclass

from typing_extensions import Literal, TypedDict

LossName = Literal["L1", "MSE", "CB", "MSSSIM"]
DataTypeName = Literal["synthetic", "real"]

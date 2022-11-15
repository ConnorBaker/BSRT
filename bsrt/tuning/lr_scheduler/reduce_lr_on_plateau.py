from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class ReduceLROnPlateauConfigSpace(ConfigSpace):
    factor: Float = loguniform(1e-4, 1.0)


@dataclass
class ReduceLROnPlateauParams(Params):
    factor: float

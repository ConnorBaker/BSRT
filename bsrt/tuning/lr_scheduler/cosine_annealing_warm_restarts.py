from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Float, Integer, loguniform, randint

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class CosineAnnealingWarmRestartsConfigSpace(ConfigSpace):
    T_0: Integer = randint(1, 100)
    T_mult: Integer = randint(1, 10)
    eta_min: Float = loguniform(1e-5, 1e-1)


@dataclass
class CosineAnnealingWarmRestartsParams(Params):
    T_0: int
    T_mult: int
    eta_min: float

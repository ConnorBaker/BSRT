from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class SGDConfigSpace(ConfigSpace):
    lr: Float = loguniform(1e-6, 1.0)
    momentum: Float = loguniform(1e-9, 1.0)
    dampening: Float = loguniform(1e-9, 1.0)
    weight_decay: Float = loguniform(1e-9, 1e-3)


@dataclass
class SGDParams(Params):
    lr: float
    momentum: float
    dampening: float
    weight_decay: float

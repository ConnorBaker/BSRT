from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class AdamWConfigSpace(ConfigSpace):
    lr: Float = loguniform(1e-6, 1e-1)
    beta_gradient: Float = loguniform(1e-3, 1.0)
    beta_square: Float = loguniform(1e-3, 1.0)
    eps: Float = loguniform(1e-8, 1e-4)
    weight_decay: Float = loguniform(1e-9, 1e-3)


@dataclass
class AdamWParams(Params):
    lr: float
    beta_gradient: float
    beta_square: float
    eps: float
    weight_decay: float

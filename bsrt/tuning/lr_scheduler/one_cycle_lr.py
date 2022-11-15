from __future__ import annotations

from dataclasses import dataclass

from syne_tune.config_space import Float, loguniform, uniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class OneCycleLRConfigSpace(ConfigSpace):
    epochs: int
    steps_per_epoch: int
    max_lr: Float = loguniform(1e-3, 1.0)
    pct_start: Float = uniform(0.1, 0.5)
    base_momentum: Float = uniform(0.5, 0.9)
    max_momentum: Float = uniform(0.9, 1.0)
    div_factor: Float = uniform(1e1, 1e3)
    final_div_factor: Float = uniform(1e3, 1e5)


@dataclass
class OneCycleLRParams(Params):
    # TODO: Aren't epochs and steps_per_epoch modified by things like gradient accumulation and
    # multiple devices (num devices * batch size = effective batch size)?
    epochs: int
    steps_per_epoch: int
    max_lr: float
    pct_start: float
    base_momentum: float
    max_momentum: float
    div_factor: float
    final_div_factor: float

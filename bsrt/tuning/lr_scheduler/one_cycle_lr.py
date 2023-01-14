from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional

from syne_tune.config_space import Float, loguniform, uniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class OneCycleLRConfigSpace(ConfigSpace):
    max_lr: Float = field(default_factory=partial(loguniform, 1e-3, 1.0))
    pct_start: Float = field(default_factory=partial(uniform, 0.1, 0.5))
    base_momentum: Float = field(default_factory=partial(uniform, 0.5, 0.9))
    max_momentum: Float = field(default_factory=partial(uniform, 0.9, 1.0))
    div_factor: Float = field(default_factory=partial(uniform, 1e1, 1e3))
    final_div_factor: Float = field(default_factory=partial(uniform, 1e3, 1e5))


@dataclass
class OneCycleLRParams(Params):
    total_steps: Optional[int]
    max_lr: float
    pct_start: float
    base_momentum: float
    max_momentum: float
    div_factor: float
    final_div_factor: float

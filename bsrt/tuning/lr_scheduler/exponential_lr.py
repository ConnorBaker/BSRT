from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class ExponentialLRConfigSpace(ConfigSpace):
    gamma: Float = field(default_factory=partial(loguniform, 1e-4, 1.0))


@dataclass
class ExponentialLRParams(Params):
    gamma: float

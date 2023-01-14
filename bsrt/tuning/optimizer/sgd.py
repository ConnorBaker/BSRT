from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class SGDConfigSpace(ConfigSpace):
    lr: Float = field(default_factory=partial(loguniform, 1e-6, 1.0))
    momentum: Float = field(default_factory=partial(loguniform, 1e-9, 1.0))
    dampening: Float = field(default_factory=partial(loguniform, 1e-9, 1.0))
    weight_decay: Float = field(default_factory=partial(loguniform, 1e-9, 1e-3))


@dataclass
class SGDParams(Params):
    lr: float
    momentum: float
    dampening: float
    weight_decay: float

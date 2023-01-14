from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from syne_tune.config_space import Float, loguniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class AdamWConfigSpace(ConfigSpace):
    lr: Float = field(default_factory=partial(loguniform, 1e-6, 1e-1))
    beta_gradient: Float = field(default_factory=partial(loguniform, 1e-3, 1.0))
    beta_square: Float = field(default_factory=partial(loguniform, 1e-3, 1.0))
    eps: Float = field(default_factory=partial(loguniform, 1e-8, 1e-4))
    weight_decay: Float = field(default_factory=partial(loguniform, 1e-9, 1e-3))


@dataclass
class AdamWParams(Params):
    lr: float
    beta_gradient: float
    beta_square: float
    eps: float
    weight_decay: float

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from syne_tune.config_space import Categorical, Float, choice, uniform

from bsrt.tuning.config_space import ConfigSpace
from bsrt.tuning.params import Params


@dataclass
class BSRTConfigSpace(ConfigSpace):
    attn_drop_rate: Float = field(default_factory=partial(uniform, 0.0, 1.0))
    drop_path_rate: Float = field(default_factory=partial(uniform, 0.0, 1.0))
    drop_rate: Float = field(default_factory=partial(uniform, 0.0, 1.0))
    mlp_ratio: Categorical = field(default_factory=partial(choice, [2**i for i in range(1, 6)]))
    flow_alignment_groups: Categorical = field(default_factory=partial(choice, [2**i for i in range(2, 5)]))
    num_features: Categorical = field(default_factory=partial(choice, [2**i for i in range(4, 8)]))
    qkv_bias: Categorical = field(default_factory=partial(choice, [True, False]))


@dataclass
class BSRTParams(Params):
    attn_drop_rate: float
    drop_path_rate: float
    drop_rate: float
    mlp_ratio: float
    flow_alignment_groups: int
    num_features: int
    qkv_bias: bool

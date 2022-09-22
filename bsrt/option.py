from dataclasses import dataclass
from typing import Tuple
from typing_extensions import Literal


@dataclass(init=False)
class Config:
    data_type: Literal["synthetic", "real"]
    n_resblocks: int
    n_feats: int
    n_colors: int
    lr: float
    burst_size: int
    burst_channel: int
    swinfeature: bool
    model_level: Literal["S", "L"]

    finetune: bool
    finetune_align: bool
    finetune_swin: bool
    finetune_conv: bool
    finetune_prelayer: bool
    finetune_upconv: bool
    finetune_spynet: bool

    seed: int
    use_checkpoint: bool

    data_dir: str
    mode: str
    scale: int
    patch_size: int
    rgb_range: int

    act: str
    res_scale: float
    shift_mean: bool
    dilation: bool

    n_resgroups: int
    reduction: int
    DA: bool
    CA: bool
    non_local: bool

    epochs: int
    batch_size: int
    gan_k: int

    decay: str
    gamma: float
    optimizer: Literal["SGD", "ADAM", "RMSprop"]
    momentum: float
    betas: Tuple[int, int]
    epsilon: float
    weight_decay: float
    gclip: float

    loss: str
    skip_threshold: float

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

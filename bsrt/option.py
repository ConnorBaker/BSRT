from dataclasses import dataclass
import dataclasses
from typing import Tuple
from typing_extensions import Literal


@dataclass
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

    seed: int
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

    loss: Literal["L1", "MSE", "CB", "MSSIM"]

    @classmethod
    def from_dict(cls, d) -> "Config":
        field_names = set(f.name for f in dataclasses.fields(cls))
        c = Config(**{k:v for k,v in d.items() if k in field_names})
        return c
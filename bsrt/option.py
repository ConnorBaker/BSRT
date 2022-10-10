from dataclasses import dataclass
from typing_extensions import Literal

LossName = Literal["L1", "MSE", "CB", "MSSSIM"]
OptimizerName = Literal["SGD", "ADAM", "RMSprop"]
DataTypeName = Literal["synthetic", "real"]

@dataclass
class Config:
    data_type: DataTypeName
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

    non_local: bool

    seed: int
    batch_size: int
    epochs: int
    gan_k: int

    decay_milestones: list[int]
    gamma: float
    optimizer: OptimizerName
    momentum: float
    beta_gradient: float
    beta_square: float
    epsilon: float
    weight_decay: float

    loss: LossName

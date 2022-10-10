from dataclasses import dataclass
from typing_extensions import Literal
from ray import tune
from operator import pow
from functools import partial
from ray.tune.search.sample import Categorical, Float

LossName = Literal["L1", "MSE", "CB", "MSSSIM"]
OptimizerName = Literal["SGD", "ADAM", "RMSprop"]
DataTypeName = Literal["synthetic", "real"]

@dataclass
class ConfigHyperTuner:
    n_resblocks: Categorical = tune.choice(list(map(partial(pow, 2), range(3, 8))))
    n_feats: Categorical = tune.choice(list(map(partial(pow, 2), range(5, 10))))
    lr: Float = tune.loguniform(1e-5, 1e-3)

    # Assumed to be true
    # swinfeature: Categorical = tune.choice([True, False])
    # use_checkpoint: Categorical = tune.choice([True, False])
    # non_local: Categorical = tune.choice([True, False])
    
    gan_k: Categorical = tune.choice(range(1, 5))
    # decay_milestones: Categorical = tune.choice(
    #     [[x, y] for x in range(40, 300, 20) for y in range(80, 400, 20) if x < y]
    # )
    gamma: Float = tune.loguniform(1e-4, 1.0)
    # optimizer: Categorical = tune.choice(["SGD", "ADAM", "RMSprop"])
    momentum: Float = tune.loguniform(1e-4, 1.0)
    beta_gradient: Float = tune.loguniform(1e-4, 1.0)
    beta_square: Float = tune.loguniform(1e-4, 1.0)
    epsilon: Float = tune.loguniform(1e-10, 1e-4)
    weight_decay: Float = tune.loguniform(1e-10, 1e-4)
    # loss: Categorical = tune.choice(["L1", "MSE", "CB", "MSSIM"])


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

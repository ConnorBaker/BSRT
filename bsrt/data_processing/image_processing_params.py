from dataclasses import dataclass


@dataclass
class ImageProcessingParams:
    """Dataclass for storing image processing parameters"""

    random_ccm: bool = True
    random_gains: bool = True
    smoothstep: bool = True
    compress_gamma: bool = True
    add_noise: bool = True

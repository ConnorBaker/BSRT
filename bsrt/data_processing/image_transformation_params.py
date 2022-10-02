from dataclasses import dataclass


@dataclass
class ImageTransformationParams:
    """Dataclass for storing transformation parameters"""

    border_crop: int = 0
    max_rotation: float = 0.0
    max_shear: float = 0.0
    max_ar_factor: float = 0.0
    max_scale: float = 0.0
    max_translation: float = 0.0

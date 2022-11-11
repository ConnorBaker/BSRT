from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor


@dataclass
class Noises:
    """Noise parameters."""

    shot_noise: float = 0.01
    read_noise: float = 0.0005

    _log_min_shot_noise: ClassVar[float] = math.log(0.0001)
    _log_max_shot_noise: ClassVar[float] = math.log(0.012)

    @staticmethod
    def random_noise_levels() -> Noises:
        """Generates random noise levels from a log-log linear distribution."""
        log_shot_noise = random.uniform(Noises._log_min_shot_noise, Noises._log_max_shot_noise)
        shot_noise = math.exp(log_shot_noise)

        log_read_noise = (2.72 * log_shot_noise + 1.14) + random.gauss(mu=0.0, sigma=0.26)
        read_noise = math.exp(log_read_noise)
        return Noises(shot_noise, read_noise)

    def apply(self, image: Tensor) -> Tensor:
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * self.shot_noise + self.read_noise
        noise = torch.FloatTensor(image.shape).normal_() * variance.sqrt()
        return image + noise

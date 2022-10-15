from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class Noises:
    """Noise parameters."""

    shot_noise: float = 0.01
    read_noise: float = 0.0005

    @staticmethod
    def random_noise_levels() -> Noises:
        """Generates random noise levels from a log-log linear distribution."""
        log_min_shot_noise = math.log(0.0001)
        log_max_shot_noise = math.log(0.012)
        log_shot_noise = random.uniform(log_min_shot_noise, log_max_shot_noise)
        shot_noise = math.exp(log_shot_noise)

        line: Callable[[float], float] = lambda x: 2.18 * x + 1.20
        log_read_noise = line(log_shot_noise) + random.gauss(mu=0.0, sigma=0.26)
        read_noise = math.exp(log_read_noise)
        return Noises(shot_noise, read_noise)

    def apply(self, image: Tensor) -> Tensor:
        """Adds random shot (proportional to image) and read (independent) noise."""
        variance = image * self.shot_noise + self.read_noise
        noise = torch.FloatTensor(image.shape).normal_() * variance.sqrt()
        return image + noise

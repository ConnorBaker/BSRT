from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RgbGains:
    """Container for gains.
    RGB gain represents brightening.
    Red and blue gains represent white balance.
    """

    rgb_gain: float
    red_gain: float
    blue_gain: float

    @staticmethod
    def random_gains() -> RgbGains:
        """Generates random gains for brightening and white balance."""
        rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)
        red_gain = random.uniform(1.9, 2.4)
        blue_gain = random.uniform(1.5, 1.9)
        return RgbGains(rgb_gain, red_gain, blue_gain)

    def apply(self, image: Tensor) -> Tensor:
        """Inverts gains while safely handling saturated pixels."""
        assert image.dim() == 3
        channels = image.shape[0]
        assert channels == 3 or channels == 4
        middle = [1.0] * (channels // 2)

        gains: Tensor = (
            torch.tensor([self.red_gain, *middle, self.blue_gain], dtype=image.dtype)
            * self.rgb_gain
        )
        gains = gains.view(-1, 1, 1)

        return (image * gains).clamp(0.0, 1.0)

    def safe_invert_gains(self, image: Tensor) -> Tensor:
        """Inverts gains while safely handling saturated pixels."""
        assert image.dim() == 3
        assert image.shape[0] == 3

        gains = torch.tensor([1.0 / self.red_gain, 1.0, 1.0 / self.blue_gain]) / self.rgb_gain
        gains = gains.view(-1, 1, 1)

        # Prevents dimming of saturated pixels by smoothly masking gains near white.
        gray = image.mean(dim=0, keepdim=True)
        inflection = 0.9
        mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection)) ** 2.0

        safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
        return image * safe_gains

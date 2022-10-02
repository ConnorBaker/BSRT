from typing_extensions import Literal

GanType = Literal["GAN", "SNGAN", "RGAN", "WGAN"]
BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
NormalizationMode = Literal["crop", "pad"]
InterpolationType = Literal["nearest", "bilinear", "bicubic", "lanczos"]

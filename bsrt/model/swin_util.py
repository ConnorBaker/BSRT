# Modified from
# https://github.com/JingyunLiang/SwinIR/blob/9d14daa8b6169c57e7604af8d0bf31f1c3496a50/models/network_swinir.py
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

from math import prod
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.swin_transformer import SwinTransformerBlockV2  # type: ignore[import]
from typing_extensions import TypeVar

_T = TypeVar("_T")


def to_2tuple(x: _T) -> tuple[_T, _T]:  # type: ignore[valid-type]
    return (x, x)


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        input_resolution: Iterable[int],
        dim: int,
        norm_layer: nn.Module = nn.LayerNorm,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default:
            True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            Default: None
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: list[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Iterable[float] = 0.0,
        norm_layer: None | nn.Module = nn.LayerNorm,  # type: ignore[assignment]
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SwinTransformerBlockV2(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0 if (i % 2 == 0) else ws // 2 for ws in window_size],
                    mlp_ratio=mlp_ratio,
                    dropout=drop,
                    attention_dropout=attn_drop,
                    stochastic_depth_prob=_drop_path,
                    norm_layer=norm_layer,  # type: ignore[arg-type]
                )
                for i, _drop_path in zip(
                    range(depth),
                    drop_path if isinstance(drop_path, Iterable) else [drop_path] * depth,
                )
            ]
        )

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:  # type: ignore[override]
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = self.blocks(x)
        x = x.view(B, L, C)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default:
            True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            Default: None
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: list[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Iterable[float] = 0.0,
        norm_layer: None | nn.Module = nn.LayerNorm,  # type: ignore[assignment]
        img_size: int = 224,
        patch_size: int = 4,
        resi_connection: str = "1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim

        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )

        self.conv: Callable[[Tensor], Tensor]
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:  # type: ignore[override]
        x = self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        return x


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: None | nn.Module = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size // patch_size] * 2
        self.num_patches = prod(self.patches_resolution)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, x: Tensor, use_norm: bool = True) -> Tensor:  # type: ignore[override]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if use_norm and self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: None | nn.Module = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size // patch_size] * 2
        self.num_patches = prod(self.patches_resolution)

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, x: Tensor, x_size: tuple[int, int]) -> Tensor:  # type: ignore[override]
        B, _HW, _C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

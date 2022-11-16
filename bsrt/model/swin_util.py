# Modified from
# https://github.com/JingyunLiang/SwinIR/blob/9d14daa8b6169c57e7604af8d0bf31f1c3496a50/models/network_swinir.py
# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

from typing import Callable, Iterable, List, Tuple, TypeVar, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.swin_transformer import SwinTransformerBlockV2

_T = TypeVar("_T")


def to_2tuple(x: _T) -> Tuple[_T, _T]:
    return (x, x)


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
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
        window_size: List[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, Iterable[float]] = 0.0,
        norm_layer=nn.LayerNorm,
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
                    norm_layer=norm_layer,
                )
                for i, _drop_path in zip(
                    range(depth),
                    drop_path if isinstance(drop_path, Iterable) else [drop_path] * depth,
                )
            ]
        )

    def forward(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
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
        window_size: List[int],
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Union[float, Iterable[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
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

    def forward(self, x: Tensor, x_size: Tuple[int, int]) -> Tensor:
        x = (
            self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size)))
            + x
        )
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        _img_size: Tuple[int, int] = to_2tuple(img_size)
        _patch_size: Tuple[int, int] = to_2tuple(patch_size)
        patches_resolution = [
            _img_size[0] // _patch_size[0],
            _img_size[1] // _patch_size[1],
        ]
        self.img_size = _img_size
        self.patch_size = _patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, use_norm=True):
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

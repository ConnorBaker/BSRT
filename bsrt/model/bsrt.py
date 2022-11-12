from dataclasses import dataclass, field
from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from bsrt.model import arch_util
from bsrt.model import swin_util as swu
from bsrt.model.cross_non_local_fusion import CrossNonLocalFusion
from bsrt.model.flow_guided_pcd_align import FlowGuidedPCDAlign
from bsrt.model.spynet_util import SpyNet
from bsrt.utils.bilinear_upsample_2d import bilinear_upsample_2d


@dataclass(eq=False)
class BSRT(nn.Module):
    """BurstSR model

    Args:
        attn_drop_rate (float): attention drop rate
        drop_path_rate (float): drop path rate
        drop_rate (float): drop rate
        flow_alignment_groups (int): number of groups for flow alignment
        in_chans (int): number of input channels
        lr (float): learning rate
        mlp_ratio (float): mlp ratio
        model_level (str): S or L for small or large model
        norm_layer (int -> nn.Module): normalization layer
        num_features (int): number of features in the feature extraction network
        num_frames (int): number of frames in the burst
        out_chans (int): number of output channels
        patch_norm (bool): patch norm
        patch_size (int): patch size
        qk_scale (float | None): qk scale
        qkv_bias (bool): qkv bias
        upscale (int): upscale
        window_size (int): window size
    """

    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    drop_rate: float = 0.0
    flow_alignment_groups: int = 8
    in_chans: int = 4  # RAW images are RGGB or the like, so 4 channels
    lr: float = 1e-4
    mlp_ratio: float = 4.0
    model_level: Literal["S", "L"] = "S"
    norm_layer: Type[nn.LayerNorm] = nn.LayerNorm
    num_features: int = 64
    num_frames: int = 14
    out_chans: int = 3  # RGB output so 3 channels
    patch_norm: bool = True
    patch_size: int = 1
    qk_scale: Union[float, None] = None
    qkv_bias: bool = True
    upscale: int = 4
    window_size: int = 7

    center: int = field(init=False, default=0)
    conv_first: nn.Conv2d = field(init=False)
    conv_flow: nn.Conv2d = field(init=False)
    depths: List[int] = field(init=False)
    embed_dim: int = field(init=False)
    flow_ps: nn.PixelShuffle = field(init=False)
    img_size: int = field(init=False)
    num_heads: List[int] = field(init=False)
    num_layers: int = field(init=False)
    patch_embed: swu.PatchEmbed = field(init=False)
    patch_unembed: swu.PatchUnEmbed = field(init=False)
    spynet: SpyNet = field(init=False)

    def __post_init__(self):
        super().__init__()

        if self.model_level == "S":
            self.depths = [6] * 1 + [6] * 4
            self.num_heads = [6] * 1 + [6] * 4
            self.embed_dim = 60
        elif self.model_level == "L":
            self.depths = [6] * 1 + [8] * 6
            self.num_heads = [6] * 1 + [6] * 6
            self.embed_dim = 180

        # TODO: In the original code, patch_size was only used for calculating img_size.
        # The rest of the uses of patch_size were using a hardcoded value of one.
        self.img_size = self.patch_size * 2
        # TODO: We set patch_size to one here manually to duplicate that behavior.
        self.patch_size = 1

        self.num_layers = len(self.depths)
        self.spynet = SpyNet([3, 4, 5]).eval().requires_grad_(False)
        self.flow_ps = nn.PixelShuffle(2)

        # split image into non-overlapping patches
        self.patch_embed = swu.PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
        )

        # merge non-overlapping patches into image
        self.patch_unembed = swu.PatchUnEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.embed_dim,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
        )

        # 1, shallow feature extraction
        self.conv_flow = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv_first = nn.Conv2d(
            self.in_chans * (1 + 2 * 0), self.embed_dim, 3, 1, 1, bias=True
        )

        # stochastic depth
        dpr: List[float] = torch.linspace(
            0, self.drop_path_rate, sum(self.depths)
        ).tolist()  # stochastic depth decay rule

        self.pre_layers = nn.ModuleList()
        for i_layer in range(self.depths[0]):
            layer = swu.SwinTransformerBlock(
                dim=self.embed_dim,
                input_resolution=(
                    self.patch_embed.patches_resolution[0] // 2,
                    self.patch_embed.patches_resolution[1] // 2,
                ),
                num_heads=self.num_heads[0],
                window_size=self.window_size,
                shift_size=0 if (i_layer % 2 == 0) else self.window_size // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[i_layer],
                norm_layer=self.norm_layer,
            )
            self.pre_layers.append(layer)

        self.pre_norm = self.norm_layer(self.embed_dim)

        self.conv_after_pre_layer = nn.Conv2d(
            self.embed_dim, self.num_features * 4, 3, 1, 1, bias=True
        )
        self.mid_ps = nn.PixelShuffle(2)

        self.fea_L2_conv1 = nn.Conv2d(self.num_features, self.num_features * 2, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(
            self.num_features * 2, self.num_features * 4, 3, 2, 1, bias=True
        )

        # 2, Feature Enhanced PCD Align

        # Top layers
        self.toplayer = nn.Conv2d(
            self.num_features * 4, self.num_features, kernel_size=1, stride=1, padding=0
        )
        # Smooth layers
        self.smooth1 = nn.Conv2d(
            self.num_features, self.num_features, kernel_size=3, stride=1, padding=1
        )
        self.smooth2 = nn.Conv2d(
            self.num_features, self.num_features, kernel_size=3, stride=1, padding=1
        )
        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            self.num_features * 2, self.num_features, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            self.num_features * 1, self.num_features, kernel_size=1, stride=1, padding=0
        )

        self.align = FlowGuidedPCDAlign(nf=self.num_features, groups=self.flow_alignment_groups)
        # 3, Multi-frame Feature Fusion
        self.fusion = CrossNonLocalFusion(
            nf=self.num_features,
            out_feat=self.embed_dim,
            nframes=self.num_frames,
            center=self.center,
        )

        # 4, deep feature extraction
        self.pos_drop = nn.Dropout(self.drop_rate)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            layer = swu.RSTB(
                dim=self.embed_dim,
                input_resolution=(
                    self.patch_embed.patches_resolution[0],
                    self.patch_embed.patches_resolution[1],
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=self.norm_layer,
                downsample=None,
                img_size=self.img_size,
                patch_size=self.patch_size,
            )
            self.layers.append(layer)

        self.norm = self.norm_layer(self.embed_dim)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(self.embed_dim, self.embed_dim, 3, 1, 1)

        # 5, high quality image reconstruction

        self.upconv1 = nn.Conv2d(self.embed_dim, self.num_features * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(self.num_features, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, self.out_chans, 3, 1, 1, bias=True)

        # skip
        self.skip_pixel_shuffle = nn.PixelShuffle(2)
        self.skipup1 = nn.Conv2d(self.in_chans // 4, self.num_features * 4, 3, 1, 1, bias=True)
        self.skipup2 = nn.Conv2d(self.num_features, self.out_chans * 4, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            swu.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        return bilinear_upsample_2d(x, scale_factor=2) + y

    def check_image_size(self, x: Tensor) -> Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def pre_forward_features(self, x: Tensor) -> Tensor:
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x, use_norm=True)
        x = self.pos_drop(x)

        for layer in self.pre_layers:
            x = layer(x, x_size)

        x = self.pre_norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # B: batch size
        # N: number of frames
        # C: number of channels
        # H: height
        # W: width
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        # skip module
        skip1 = self.lrelu2(
            self.skip_pixel_shuffle(self.skipup1(self.skip_pixel_shuffle(x_center)))
        )
        skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))

        x_ = self.conv_flow(self.flow_ps(x.view(B * N, C, H, W))).view(B, N, -1, H * 2, W * 2)

        # calculate flows
        ref_flows = self.get_ref_flows(x_)

        # extract LR features
        x = self.lrelu(self.conv_first(x.view(B * N, -1, H, W)))

        L1_fea = self.mid_ps(self.conv_after_pre_layer(self.pre_forward_features(x)))
        _, _, H, W = L1_fea.size()

        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))

        # FPN enhance features
        L3_fea = self.lrelu(self.toplayer(L3_fea))
        L2_fea = self.smooth1(self._upsample_add(L3_fea, self.latlayer1(L2_fea)))
        L1_fea = self.smooth2(self._upsample_add(L2_fea, self.latlayer2(L1_fea)))

        L1_fea = L1_fea.view(B, N, -1, H, W).contiguous()
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2).contiguous()
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4).contiguous()

        # PCD align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(),
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone(),
        ]
        _aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(),
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone(),
            ]
            flows_l = [
                ref_flows[0][:, i, :, :, :].clone(),
                ref_flows[1][:, i, :, :, :].clone(),
                ref_flows[2][:, i, :, :, :].clone(),
            ]
            # print(nbr_fea_l[0].shape, flows_l[0].shape)
            nbr_warped_l = [
                arch_util.flow_warp(nbr_fea_l[0], flows_l[0].permute(0, 2, 3, 1)),
                arch_util.flow_warp(nbr_fea_l[1], flows_l[1].permute(0, 2, 3, 1)),
                arch_util.flow_warp(nbr_fea_l[2], flows_l[2].permute(0, 2, 3, 1)),
            ]
            _aligned_fea.append(self.align(nbr_fea_l, nbr_warped_l, ref_fea_l, flows_l))

        aligned_fea = torch.stack(_aligned_fea, dim=1)  # [B, N, C, H, W] --> [B, T, C, H, W]

        x = self.lrelu(self.fusion(aligned_fea))

        x = self.lrelu(self.conv_after_body(self.forward_features(x))) + x

        x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
        x = skip1 + x
        x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        x = self.lrelu(self.HRconv(x))
        x = self.conv_last(x)

        x = skip2 + x

        clamped = x.clamp(0, 1)
        return clamped

    def get_ref_flows(self, x: Tensor) -> List[Tensor]:
        """Get flow between frames ref and other"""
        b, n, c, h, w = x.size()
        x_nbr = x.reshape(-1, c, h, w)
        x_ref = (
            x[:, self.center : self.center + 1, :, :, :].repeat(1, n, 1, 1, 1).reshape(-1, c, h, w)
        )

        # backward
        flows: Tensor = self.spynet(x_ref, x_nbr)
        flows_list = [
            flow.view(b, n, 2, h // (2 ** (i)), w // (2 ** (i)))
            for flow, i in zip(flows, range(3))
        ]

        return flows_list

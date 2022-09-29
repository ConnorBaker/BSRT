from model.cross_non_local_fusion import CrossNonLocalFusion
from model.flow_guided_pcd_align import FlowGuidedPCDAlign
from model.spynet_util import SpyNet
from option import Config
from torch.nn.parameter import Parameter
from utils.bilinear_upsample_2d import bilinear_upsample_2d
import functools
import model.arch_util as arch_util
import model.swin_util as swu
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSRT(nn.Module):
    def __init__(
        self,
        config: Config,
        nframes=8,
        img_size=64,
        patch_size=1,
        in_chans=3,
        out_chans=3,
        embed_dim=96,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_swin_checkpoint=False,
        upscale=4,
        non_local=False,
        **kwargs,
    ):
        super(BSRT, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        num_feat = 64
        groups = 8
        # embed_dim = num_feat
        back_RBs = 5
        n_resblocks = 6

        self.config = config
        self.center = 0
        self.upscale = upscale
        self.window_size = window_size
        self.non_local = non_local
        self.nframes = nframes
        self.batch_size = config.batch_size
        self.loss_name = config.loss

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.spynet = SpyNet([3, 4, 5])
        self.conv_flow = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.flow_ps = nn.PixelShuffle(2)

        # split image into non-overlapping patches
        self.patch_embed = swu.PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = swu.PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(
            num_in_ch * (1 + 2 * 0), embed_dim, 3, 1, 1, bias=True
        )

        # # stochastic depth
        dpr = torch.linspace(
            0, drop_path_rate, sum(depths)
        ).tolist()  # stochastic depth decay rule

        if config.swinfeature:
            print("using swinfeature")
            self.pre_layers = nn.ModuleList()
            for i_layer in range(depths[0]):
                layer = swu.SwinTransformerBlock(
                    dim=embed_dim,
                    input_resolution=(
                        patches_resolution[0] // 2,
                        patches_resolution[1] // 2,
                    ),
                    num_heads=num_heads[0],
                    window_size=window_size,
                    shift_size=0 if (i_layer % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i_layer],
                    norm_layer=norm_layer,
                )
                self.pre_layers.append(layer)

            self.pre_norm = norm_layer(embed_dim)
        else:
            WARB = functools.partial(arch_util.WideActResBlock, nf=embed_dim)
            self.feature_extraction = arch_util.make_layer(WARB, 5)

        self.conv_after_pre_layer = nn.Conv2d(
            embed_dim, num_feat * 4, 3, 1, 1, bias=True
        )
        self.mid_ps = nn.PixelShuffle(2)

        self.fea_L2_conv1 = nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=True)

        #####################################################################################################
        ################################### 2, Feature Enhanced PCD Align ###################################

        # Top layers
        self.toplayer = nn.Conv2d(
            num_feat * 4, num_feat, kernel_size=1, stride=1, padding=0
        )
        # Smooth layers
        self.smooth1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            num_feat * 2, num_feat, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            num_feat * 1, num_feat, kernel_size=1, stride=1, padding=0
        )

        # self.align = PCD_Align(nf=num_feat, groups=groups)
        self.align = FlowGuidedPCDAlign(nf=num_feat, groups=groups)
        #####################################################################################################
        ################################### 3, Multi-frame Feature Fusion  ##################################

        if self.non_local:
            print("using non_local")
            self.fusion = CrossNonLocalFusion(
                nf=num_feat, out_feat=embed_dim, nframes=nframes, center=self.center
            )
        else:
            self.fusion = nn.Conv2d(nframes * num_feat, embed_dim, 1, 1, bias=True)

        #####################################################################################################
        ################################### 4, deep feature extraction ######################################

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(torch.zeros(1, num_patches, embed_dim))
            swu.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(1, self.num_layers):
            layer = swu.RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_swin_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################ 5, high quality image reconstruction ################################

        self.upconv1 = nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, config.n_colors, 3, 1, 1, bias=True)

        #### skip #############
        self.skip_pixel_shuffle = nn.PixelShuffle(2)
        self.skipup1 = nn.Conv2d(num_in_ch // 4, num_feat * 4, 3, 1, 1, bias=True)
        self.skipup2 = nn.Conv2d(num_feat, config.n_colors * 4, 3, 1, 1, bias=True)

        #### activation function
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def _upsample_add(self, x, y):
        return bilinear_upsample_2d(x, scale_factor=2) + y

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def pre_forward_features(self, x):
        if self.config.swinfeature:
            x_size = (x.shape[-2], x.shape[-1])
            x = self.patch_embed(x, use_norm=True)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            for idx, layer in enumerate(self.pre_layers):
                x = layer(x, x_size)

            x = self.pre_norm(x)
            x = self.patch_unembed(x, x_size)

        else:
            x = self.feature_extraction(x)

        return x

    def forward_features(self, x):
        x_size = (x.shape[-2], x.shape[-1])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            x = layer(x, x_size)
            if torch.any(torch.isinf(x)) or torch.any(torch.isnan(x)):
                print("layer: ", idx)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        # B: batch size
        # N: number of frames
        # C: number of channels
        # H: height
        # W: width
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### skip module ########
        skip1 = self.lrelu2(
            self.skip_pixel_shuffle(self.skipup1(self.skip_pixel_shuffle(x_center)))
        )
        skip2 = self.skip_pixel_shuffle(self.skipup2(skip1))

        x_ = self.conv_flow(self.flow_ps(x.view(B * N, C, H, W))).view(
            B, N, -1, H * 2, W * 2
        )

        # calculate flows
        ref_flows = self.get_ref_flows(x_)

        #### extract LR features
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

        #### PCD align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(),
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone(),
        ]
        aligned_fea = []
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
                arch_util.flow_warp(
                    nbr_fea_l[0], flows_l[0].permute(0, 2, 3, 1), "bilinear"
                ),
                arch_util.flow_warp(
                    nbr_fea_l[1], flows_l[1].permute(0, 2, 3, 1), "bilinear"
                ),
                arch_util.flow_warp(
                    nbr_fea_l[2], flows_l[2].permute(0, 2, 3, 1), "bilinear"
                ),
            ]
            aligned_fea.append(self.align(nbr_fea_l, nbr_warped_l, ref_fea_l, flows_l))

        aligned_fea = torch.stack(
            aligned_fea, dim=1
        )  # [B, N, C, H, W] --> [B, T, C, H, W]

        if not self.non_local:
            aligned_fea = aligned_fea.view(B, -1, H, W)

        x = self.lrelu(self.fusion(aligned_fea))

        x = self.lrelu(self.conv_after_body(self.forward_features(x))) + x

        x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
        x = skip1 + x
        x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        x = self.lrelu(self.HRconv(x))
        x = self.conv_last(x)

        x = skip2 + x
        return x

    def get_ref_flows(self, x):
        """Get flow between frames ref and other"""
        b, n, c, h, w = x.size()
        x_nbr = x.reshape(-1, c, h, w)
        x_ref = (
            x[:, self.center : self.center + 1, :, :, :]
            .repeat(1, n, 1, 1, 1)
            .reshape(-1, c, h, w)
        )

        # backward
        flows = self.spynet(x_ref, x_nbr)
        flows_list = [
            flow.view(b, n, 2, h // (2 ** (i)), w // (2 ** (i)))
            for flow, i in zip(flows, range(3))
        ]

        return flows_list

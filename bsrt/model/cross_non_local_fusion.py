from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import Literal


class _NonLocalBlockNDGeneral(nn.Module):
    def __init__(
        self,
        kind: Literal["concatenation", "gaussian", "embedded_gaussian", "dot_product", "cross_dot_product"],
        in_channels: int,
        inter_channels: int,
        dimension: Literal[1, 2, 3] = 3,
        sub_sample: bool = True,
        bn_layer: bool = True,
    ):
        super().__init__()

        assert dimension in [1, 2, 3]

        self.kind: Literal["concatenation", "gaussian", "embedded_gaussian", "dot_product", "cross_dot_product"] = kind
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        kernel_size: Literal[2, 4]
        if kind == "concatenation" or kind == "gaussian" or kind == "embedded_gaussian":
            kernel_size = 2
        elif kind == "dot_product" or kind == "cross_dot_product":
            kernel_size = 4
        else:
            raise ValueError(f"Unknown kind: {kind}")

        conv_nd: Callable[..., nn.Conv1d | nn.Conv2d | nn.Conv3d]
        max_pool_layer: nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d
        bn: Callable[..., nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d]
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, kernel_size, kernel_size))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(kernel_size, kernel_size))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(kernel_size))
            bn = nn.BatchNorm1d

        self.g: nn.Module = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            assert isinstance(self.W[1].weight, Tensor)
            nn.init.constant_(self.W[1].weight, 0)
            assert isinstance(self.W[1].bias, Tensor)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            # Pyright warns us that we cannot redefine self.W because it is capitalized, but
            # we capitalize it to match the math in the paper, not because it is a constant.
            self.W = conv_nd(  # type: ignore[assignment]
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            assert isinstance(self.W.weight, Tensor)
            nn.init.constant_(self.W.weight, 0)
            assert isinstance(self.W.bias, Tensor)
            nn.init.constant_(self.W.bias, 0)

        if kind != "gaussian":
            self.theta = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

            self.phi: nn.Module = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        if kind == "concatenation":
            self.concat_project = nn.Sequential(nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False), nn.ReLU())

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer) if kind != "gaussian" else max_pool_layer

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(  # type: ignore[override]
        self, x: Tensor, ref: None | Tensor = None, return_nl_map: bool = False
    ) -> Tensor | Sequence[Tensor]:
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        assert (ref is not None) == (self.kind == "cross_dot_product")
        batch_size = x.size(0)

        g_x: Tensor = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # Calculate phi_x
        phi_x: Tensor
        if self.kind == "gaussian" and not self.sub_sample:
            phi_x = x.view(batch_size, self.in_channels, -1)
        else:
            # In this case, we always need to calculate self.phi(x)
            phi_x = self.phi(x)
            if self.kind == "gaussian":
                phi_x = phi_x.view(batch_size, self.in_channels, -1)
            elif self.kind == "concatenation":
                # (b, c, 1, N)
                phi_x = phi_x.view(batch_size, self.inter_channels, 1, -1)
            else:
                phi_x = phi_x.view(batch_size, self.inter_channels, -1)

        # Calculate theta_x (or theta_ref)
        theta_x: Tensor
        if self.kind == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
        else:
            # In this case, we always need to calculate self.theta(x) (or self.theta(ref))
            # ref is only not None when self.kind == "cross_dot_product"
            theta_x = self.theta(x if ref is None else ref)
            if self.kind == "concatenation":
                # (b, c, N, 1)
                theta_x = theta_x.view(batch_size, self.inter_channels, -1, 1)
            else:
                theta_x = theta_x.view(batch_size, self.inter_channels, -1)

        # Calculate f_div_C
        if self.kind == "concatenation":
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat_feature = torch.cat([theta_x, phi_x], dim=1)
            f: Tensor = self.concat_project(concat_feature)
            b, _, h, w = f.size()
            f = f.view(b, h, w)
            n = f.size(-1)
            f_div_C = f / n
        else:
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)
            if self.kind == "gaussian" or self.kind == "embedded_gaussian":
                f_div_C = F.softmax(f, dim=-1)
            else:
                n = f.size(-1)
                f_div_C = f / n

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y: Tensor = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class CrossNonLocalFusion(nn.Module):
    """Cross Non Local fusion module"""

    def __init__(self, nf: int = 64, out_feat: int = 96, nframes: int = 5, center: int = 2):
        super(CrossNonLocalFusion, self).__init__()
        self.center = center

        self.non_local_T = nn.ModuleList()
        self.non_local_F = nn.ModuleList()

        for _ in range(nframes):
            self.non_local_T.append(
                _NonLocalBlockNDGeneral(
                    kind="cross_dot_product",
                    in_channels=nf,
                    inter_channels=nf // 2,
                    dimension=2,
                    sub_sample=True,
                    bn_layer=False,
                )
            )
            self.non_local_F.append(
                _NonLocalBlockNDGeneral(
                    kind="dot_product",
                    in_channels=nf,
                    inter_channels=nf // 2,
                    dimension=2,
                    sub_sample=True,
                    bn_layer=False,
                )
            )

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf * 2, out_feat, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # Pyright says we're missing *args and **kwargs here, but we're not.
    def forward(self, aligned_fea: Tensor) -> Tensor:  # type: ignore[override]
        _B, N, _C, _H, _W = aligned_fea.size()  # N video frames
        ref = aligned_fea[:, self.center, :, :, :].clone()

        cor_l: list[Tensor] = []
        non_l: list[Tensor] = []
        for i in range(N):
            nbr = aligned_fea[:, i, :, :, :]
            non_l.append(self.non_local_F[i](nbr))
            cor_l.append(self.non_local_T[i](nbr, ref))

        aligned_fea_T = torch.cat(cor_l, dim=1)
        aligned_fea_F = torch.cat(non_l, dim=1)
        aligned_fea = torch.cat([aligned_fea_T, aligned_fea_F], dim=1)

        # fusion
        fea: Tensor = self.fea_fusion(aligned_fea)

        return fea

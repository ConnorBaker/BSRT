from typing import Literal, Optional

from torch import Tensor, nn


# TODO:
# - [ ] Is inter_channels really optional? Unless a tensor's view method allows None, it's not.
# - [ ] Implement a forward method and remove the need for all the other classes to implement it.
class _NonLocalBlockNDGeneral(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inter_channels: Optional[int] = None,
        dimension: int = 3,
        sub_sample: bool = True,
        bn_layer: bool = True,
        kind: Literal[
            "concatenation", "gaussian", "embedded_gaussian", "dot_product", "cross_dot_product"
        ] = "embedded_gaussian",
    ):
        super().__init__()

        assert dimension in [1, 2, 3]

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

        self.g = conv_nd(
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
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            assert self.W.bias is not None
            nn.init.constant_(self.W.bias, 0)

        if kind != "gaussian":
            self.theta = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

            self.phi = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        if kind == "concatenation":
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False), nn.ReLU()
            )

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = (
                nn.Sequential(self.phi, max_pool_layer) if kind != "gaussian" else max_pool_layer
            )

    def forward(self, x: Tensor, return_nl_map: bool = False) -> Tensor:
        raise NotImplementedError

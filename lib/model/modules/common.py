from typing import Optional
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

Norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
Act = partial(nn.ReLU, inplace=True)
Upsample = partial(F.interpolate, mode="bilinear", align_corners=False)


def init_weights(module: nn.Module) -> None:
    """Initialize model weights."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_out")
            nn.init.zeros_(m.bias)


class ConvBN(nn.Sequential):
    """
    Sequential module for convolution and batch normalization.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int, optional): kernel size. Defaults to 3.
        stride (int, optional): stride. Defaults to 1.
        padding (int, optional): padding. Defaults to None.
        groups (int, optional): groups. Defaults to 1.
        bias (bool, optional): bias. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Initialize ConvBN module."""
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module("bn", Norm2d(out_channels))


class ConvBNReLU(ConvBN):
    """
    Sequential module for convolution, batch normalization and ReLU.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int, optional): kernel size. Defaults to 3.
        stride (int, optional): stride. Defaults to 1.
        padding (int, optional): padding. Defaults to None.
        groups (int, optional): groups. Defaults to 1.
        bias (bool, optional): bias. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Initialize ConvBNReLU module."""
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, bias)
        self.add_module("relu", Act())


class BNReLUConv(nn.Sequential):
    """
    Sequential module for batch normalization, ReLU and convolution.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int, optional): kernel size. Defaults to 3.
        stride (int, optional): stride. Defaults to 1.
        padding (int, optional): padding. Defaults to None.
        groups (int, optional): groups. Defaults to 1.
        bias (bool, optional): bias. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        """Initialize BNReLUConv module."""
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.add_module("bn", Norm2d(in_channels))
        self.add_module("relu", Act())
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=bias,
            ),
        )


class BasicBlock(nn.Module):
    """Basic residual block for ResNetV2."""

    expansion_factor: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """Initialize BasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            downsample (Optional[nn.Module], optional): Downsample layer. Defaults to None.
        """
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBN(out_channels, out_channels * self.expansion_factor, 3, 1)
        self.downsample = downsample
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNetV2."""

    expansion_factor: int = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Bottleneck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            downsample (Optional[nn.Module], optional): Downsample layer. Defaults to None.
        """
        super().__init__()
        mid_channels = out_channels // self.expansion_factor

        self.conv1 = ConvBNReLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNReLU(mid_channels, mid_channels, 3, stride)
        self.conv3 = ConvBN(mid_channels, out_channels * self.expansion_factor, 1)
        self.downsample = downsample
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out
import torch
from torch import nn

from . import common as cm


class ClassificationHead(nn.Module):
    """
    Module for classification and segmentation head.

    Args:
        in_channels (int): Number of input channels.
        head_channels (int): Number of channels in the head.
        num_classes (int): Number of classes.

    Attributes:
        classifier (nn.Sequential): Sequential module for classification head.

    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_classes: int,
        scale_factor: int = 8,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.classifier = nn.Sequential(
            cm.BNReLUConv(in_channels, head_channels, kernel_size=3),
            cm.BNReLUConv(head_channels, num_classes, kernel_size=1, bias=True),
        )

        # self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        # if self.scale_factor is not None:
        #     return cm.Upsample(self.classifier(x), scale_factor=self.scale_factor)

        return self.classifier(x)


class CenternessHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_classes: int,
        scale_factor: int = 8,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.centerness = nn.Sequential(
            cm.BNReLUConv(in_channels, head_channels, kernel_size=3),
            cm.BNReLUConv(head_channels, num_classes, kernel_size=1, bias=True),
        )

        self.centerness[-1][2].bias.data.fill_(-2.19)

        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        if self.scale_factor is not None:
            return cm.Upsample(self.centerness(x), scale_factor=self.scale_factor)

        return self.centerness(x)


class RegressionHead(nn.Module):
    """
    Module for regression head.

    Args:
        in_channels (int): Number of input channels.
        head_channels (int): Number of channels in the head.

    Attributes:
        regressor (nn.Sequential): Sequential module for regression head.

    """

    def __init__(
        self, in_channels: int, head_channels: int, scale_factor: int = 8
    ) -> None:
        """Initialize the module."""
        super().__init__()

        self.regressor = nn.Sequential(
            cm.BNReLUConv(in_channels, head_channels, kernel_size=3),
            cm.BNReLUConv(head_channels, 2, kernel_size=1, bias=True),
        )

        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """

        if self.scale_factor is not None:
            return cm.Upsample(self.regressor(x), scale_factor=self.scale_factor)

        return self.regressor(x)

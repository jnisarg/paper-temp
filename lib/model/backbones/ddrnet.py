import torch
from torch import nn

from lib.model.modules import common as cm


class DAPPM(nn.Module):
    """
    Deep Aggregation Pyramid Pooling Module used in DDRNet.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of intermediate channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        """
        Initializes the module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.scale0 = cm.BNReLUConv(in_channels, mid_channels, 1)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )

        self.process1 = cm.BNReLUConv(mid_channels, mid_channels, 3)
        self.process2 = cm.BNReLUConv(mid_channels, mid_channels, 3)
        self.process3 = cm.BNReLUConv(mid_channels, mid_channels, 3)
        self.process4 = cm.BNReLUConv(mid_channels, mid_channels, 3)

        self.compression = cm.BNReLUConv(mid_channels * 5, out_channels, 1)
        self.shortcut = cm.BNReLUConv(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        scale0 = self.scale0(x)
        scale1 = cm.Upsample(self.scale1(x), scale_factor=2) + scale0
        scale2 = cm.Upsample(self.scale2(x), scale_factor=4) + scale1
        scale3 = cm.Upsample(self.scale3(x), scale_factor=8) + scale2
        scale4 = cm.Upsample(self.scale4(x), scale_factor=(int(x.size(2)), int(x.size(3)))) + scale3

        process1 = self.process1(scale1)
        process2 = self.process2(scale2)
        process3 = self.process3(scale3)
        process4 = self.process4(scale4)

        cat_features = torch.cat([scale0, process1, process2, process3, process4], dim=1)
        out = self.compression(cat_features) + self.shortcut(x)

        return out


class PAPPM(nn.Module):
    """
    Parallel Aggregation Pyramid Pooling Module used in PIDNet.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of intermediate channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        """
        Initializes the module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.scale0 = cm.BNReLUConv(in_channels, mid_channels, 1)
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )

        self.process = cm.BNReLUConv(mid_channels * 4, mid_channels * 4, 3, groups=4)

        self.compression = cm.BNReLUConv(mid_channels * 5, out_channels, 1)
        self.shortcut = cm.BNReLUConv(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        scale0 = self.scale0(x)
        scale1 = cm.Upsample(self.scale1(x), scale_factor=2) + scale0
        scale2 = cm.Upsample(self.scale2(x), scale_factor=4) + scale0
        scale3 = cm.Upsample(self.scale3(x), scale_factor=8) + scale0
        scale4 = cm.Upsample(self.scale4(x), scale_factor=(int(x.size(2)), int(x.size(3)))) + scale0

        process = self.process(torch.cat([scale1, scale2, scale3, scale4], dim=1))

        cat_features = torch.cat([scale0, process], dim=1)
        out = self.compression(cat_features) + self.shortcut(x)

        return out


class CE(nn.Module):
    """
    Context Embedding Module.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int): Number of intermediate channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        """
        Initializes the module.
        """
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            cm.BNReLUConv(in_channels, mid_channels, 1),
        )

        self.process = cm.BNReLUConv(mid_channels, out_channels, 3)

        self.shortcut = cm.BNReLUConv(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        scale = cm.Upsample(self.scale(x), scale_factor=(int(x.size(2)), int(x.size(3))))

        process = self.process(scale)

        out = process + self.shortcut(x)

        return out


ppm_hub: dict = {
    "dappm": DAPPM,
    "pappm": PAPPM,
    "ce": CE,
}


class DDRNet(nn.Module):
    """
    Headless implementation of the DDRNet Segmentation model.

    Args:
        ppm_block (str, optional): The PPM block to use. Defaults to "DAPPM".
        planes (int, optional): The number of planes. Defaults to 32.
        ppm_planes (int, optional): The number of PPM planes. Defaults to 128.

    Attributes:
        ppm_block (str): The PPM block to use.
        planes (int): The number of planes.
        ppm_planes (int): The number of PPM planes.
        stem (nn.Sequential): The stem of the model.
        layer1 (nn.Sequential): The first layer of residual blocks.
        layer2 (nn.Sequential): The second layer of residual blocks.
        context3 (nn.Sequential): The third layer of context blocks.
        detail3 (nn.Sequential): The third layer of detail blocks.
        down3 (nn.Module): The downsampling module for context3.
        compression3 (nn.Module): The compression module for detail3.
        context4 (nn.Sequential): The fourth layer of context blocks.
        detail4 (nn.Sequential): The fourth layer of detail blocks.
        down4 (nn.Module): The downsampling module for context4.
        compression4 (nn.Module): The compression module for detail4.
        context5 (nn.Sequential): The fifth layer of context blocks.
        detail5 (nn.Sequential): The fifth layer of detail blocks.
        ppm (nn.Module): The PPM module.

    Methods:
        forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            Forward pass of the model.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: The output tensor, followed by
                intermediate tensors.

        _make_layer(self, block: nn.Module, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
            Create a layer of residual blocks.

            Args:
                block (nn.Module): The block to use.
                in_channels (int): The number of input channels.
                out_channels (int): The number of output channels.
                num_blocks (int): The number of blocks.
                stride (int): The stride.

            Returns:
                nn.Sequential: The layer of residual blocks.
    """

    def __init__(
        self,
        ppm_block: str = "DAPPM",
        planes: int = 32,
        ppm_planes: int = 128,
    ) -> None:
        super().__init__()

        self.ppm_block = ppm_block

        self.planes = planes
        self.ppm_planes = ppm_planes

        self.stem = nn.Sequential(
            cm.ConvBNReLU(3, planes, 3, stride=2, padding=1),
            cm.ConvBNReLU(planes, planes, 3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(cm.BasicBlock, planes, planes, 2, 1)
        self.layer2 = self._make_layer(cm.BasicBlock, planes, planes * 2, 2, 2)

        # Context Branch Layers
        self.context3 = self._make_layer(cm.BasicBlock, planes * 2, planes * 4, 2, 2)
        self.context4 = self._make_layer(cm.BasicBlock, planes * 4, planes * 8, 1, 2)
        self.context5 = self._make_layer(cm.Bottleneck, planes * 8, planes * 8, 1, 2)

        self.compression3 = cm.ConvBNReLU(planes * 4, planes * 2, 1, 1)
        self.compression4 = cm.ConvBNReLU(planes * 8, planes * 2, 1, 1)

        # Detail Branch Layers
        self.detail3 = self._make_layer(cm.BasicBlock, planes * 2, planes * 2, 2, 1)
        self.detail4 = self._make_layer(cm.BasicBlock, planes * 2, planes * 2, 1, 1)
        self.detail5 = self._make_layer(cm.Bottleneck, planes * 2, planes * 2, 1, 1)

        self.down3 = cm.ConvBNReLU(planes * 2, planes * 4, 3, 2)
        self.down4 = nn.Sequential(
            cm.ConvBNReLU(planes * 2, planes * 4, 3, 2),
            cm.ConvBNReLU(planes * 4, planes * 8, 3, 2),
        )

        self.ppm = ppm_hub.get(ppm_block, "ce")(planes * 16, ppm_planes, planes * 4)

        self.out_channels = [planes * 4, planes * 4, [planes * 2, planes * 2]]

    def _make_layer(
        self,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Create a layer of residual blocks.

        Args:
            block (nn.Module): The block to use.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            num_blocks (int): The number of blocks.
            stride (int): The stride.

        Returns:
            nn.Sequential: The layer of residual blocks.
        """
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion_factor:
            downsample = cm.ConvBNReLU(
                in_channels, out_channels * block.expansion_factor, 1, stride
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion_factor
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: The output tensor, followed by
            intermediate tensors.
        """
        stem = self.stem(x)

        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)

        context3 = self.context3(layer2)
        detail3 = self.detail3(layer2)

        down3 = context3 + self.down3(detail3)
        compression3 = detail3 + cm.Upsample(self.compression3(context3), scale_factor=2)

        context4 = self.context4(down3)
        detail4 = self.detail4(compression3)

        down4 = context4 + self.down4(detail4)
        compression4 = detail4 + cm.Upsample(self.compression4(context4), scale_factor=4)

        context5 = self.context5(down4)
        detail5 = self.detail5(compression4)

        ppm = cm.Upsample(self.ppm(context5), scale_factor=8)

        return ppm, detail5, (compression3, compression4)
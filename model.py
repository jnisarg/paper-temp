import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


Norm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
Act = partial(nn.ReLU, inplace=True)
Upsample = partial(F.interpolate, mode="bilinear", align_corners=False)


class BasicBlock(nn.Module):
    """Basic residual block for ResNetV2."""

    expansion_factor: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBN(out_channels, out_channels * self.expansion_factor, 3, 1)
        self.downsample = downsample
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        downsample=None,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // self.expansion_factor

        self.conv1 = ConvBNReLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNReLU(mid_channels, mid_channels, 3, stride)
        self.conv3 = ConvBN(mid_channels, out_channels * self.expansion_factor, 1)
        self.downsample = downsample
        self.act = Act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class ConvBN(nn.Sequential):

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
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, groups, bias
        )
        self.add_module("relu", Act())


class CE(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, mid_channels, 1),
        )

        self.process = ConvBNReLU(mid_channels, out_channels, 3)

        self.shortcut = ConvBNReLU(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = Upsample(self.scale(x), scale_factor=(int(x.size(2)), int(x.size(3))))

        process = self.process(scale)

        out = process + self.shortcut(x)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.planes = 32
        self.ppm_planes = 128

        self.stem = nn.Sequential(
            ConvBNReLU(3, self.planes, 3, stride=2, padding=1),
            ConvBNReLU(self.planes, self.planes, 3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(BasicBlock, self.planes, self.planes, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, self.planes, self.planes * 2, 2, 2)

        # Context Branch Layers
        self.context3 = self._make_layer(
            BasicBlock, self.planes * 2, self.planes * 4, 2, 2
        )
        self.context4 = self._make_layer(
            BasicBlock, self.planes * 4, self.planes * 8, 1, 2
        )
        self.context5 = self._make_layer(
            Bottleneck, self.planes * 8, self.planes * 8, 1, 2
        )

        self.compression3 = ConvBNReLU(self.planes * 4, self.planes * 2, 1, 1)
        self.compression4 = ConvBNReLU(self.planes * 8, self.planes * 2, 1, 1)

        # Detail Branch Layers
        self.detail3 = self._make_layer(
            BasicBlock, self.planes * 2, self.planes * 2, 2, 1
        )
        self.detail4 = self._make_layer(
            BasicBlock, self.planes * 2, self.planes * 2, 1, 1
        )
        self.detail5 = self._make_layer(
            Bottleneck, self.planes * 2, self.planes * 2, 1, 1
        )

        self.down3 = ConvBNReLU(self.planes * 2, self.planes * 4, 3, 2)
        self.down4 = nn.Sequential(
            ConvBNReLU(self.planes * 2, self.planes * 4, 3, 2),
            ConvBNReLU(self.planes * 4, self.planes * 8, 3, 2),
        )

        self.ppm = CE(self.planes * 16, self.ppm_planes, self.planes * 4)

        self.out_channels = [
            self.planes * 4,
            self.planes * 4,
            self.planes * 2,
            [self.planes * 4, self.planes * 8, self.planes * 16],
        ]

    def _make_layer(
        self,
        block: nn.Module,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion_factor:
            downsample = ConvBNReLU(
                in_channels, out_channels * block.expansion_factor, 1, stride
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion_factor
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        stem = self.stem(x)

        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)

        context3 = self.context3(layer2)
        detail3 = self.detail3(layer2)

        down3 = context3 + self.down3(detail3)
        compression3 = detail3 + Upsample(self.compression3(context3), scale_factor=2)

        context4 = self.context4(down3)
        detail4 = self.detail4(compression3)

        down4 = context4 + self.down4(detail4)
        compression4 = detail4 + Upsample(self.compression4(context4), scale_factor=4)

        context5 = self.context5(down4)
        detail5 = self.detail5(compression4)

        ppm = Upsample(self.ppm(context5), scale_factor=8)

        return ppm, detail5, compression3, [context3, context4, context5]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.feature_extractor = timm.create_model(
        #     "resnet18d",
        #     pretrained=False,
        #     features_only=True,
        #     out_indices=[1, 2, 3, 4],
        # )

        # self.feature_channels = self.feature_extractor.feature_info.channels()
        # print(self.feature_channels)

        self.encoder = Encoder()

        enc_out_channels = self.encoder.out_channels

        self.c5 = nn.Sequential(
            nn.Conv2d(enc_out_channels[3][2], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(enc_out_channels[3][1], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(enc_out_channels[3][0], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.c1 = nn.Sequential(
        #     nn.Conv2d(enc_out_channels[0], 128, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )

        self.centerness = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, kernel_size=1),
        )

        self.regression = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(enc_out_channels[0], 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 19, kernel_size=1),
        )

        self.compression3 = nn.Sequential(
            nn.Conv2d(enc_out_channels[2], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 19, kernel_size=1),
        )

        for param in self.centerness.parameters():
            param.requires_grad = False

        for param in self.regression.parameters():
            param.requires_grad = False

    def forward(self, x):
        ppm, detail5, compression3, [context3, context4, context5] = self.encoder(x)

        c5 = F.interpolate(
            self.c5(context5), scale_factor=2, mode="bilinear", align_corners=False
        )
        c4 = F.interpolate(
            self.c4(context4) + c5,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        c3 = F.interpolate(
            self.c3(context3) + c4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # print(c5.shape, c4.shape, c3.shape, ppm.shape, detail5.shape)
        # c1 = self.c1(features[0]) + c2
        classifier = F.interpolate(
            self.classifier(ppm + detail5),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )
        compression3 = F.interpolate(
            self.compression3(compression3),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )

        centerness = F.interpolate(
            self.centerness(c3 + detail5),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        regression = F.interpolate(
            self.regression(c3 + detail5),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )

        return classifier, compression3, centerness, regression


if __name__ == "__main__":
    bk = Encoder()
    bk.eval()

    x = torch.randn(1, 3, 384, 768)

    ppm, detail5, compression3, [context3, context4, context5] = bk(x)

    print(
        ppm.shape,
        detail5.shape,
        compression3.shape,
        context3.shape,
        context4.shape,
        context5.shape,
    )

    parameters = sum(p.numel() for p in bk.parameters() if p.requires_grad)
    print("parameters: ", parameters / 1e6, "M")

    model = Model()
    model.eval()

    classifier, compression3, centerness, regression = model(x)

    print(classifier.shape, compression3.shape, centerness.shape, regression.shape)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters: ", parameters / 1e6, "M")

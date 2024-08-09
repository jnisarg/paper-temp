import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        bn_eps=1e-3,
        bn_momentum=0.01,
    ):
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
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
        )


class ConvBNReLU(ConvBN):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        bn_eps=1e-3,
        bn_momentum=0.01,
        relu6=False,
        inplace=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias,
            bn_eps,
            bn_momentum,
        )

        self.add_module(
            "relu", nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=inplace)
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, relu6=False
    ):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBN(
            out_channels, out_channels * self.expansion, kernel_size=3, stride=1
        )

        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, relu6=False
    ):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBNReLU(
            out_channels, out_channels, kernel_size=3, stride=stride
        )
        self.conv3 = ConvBN(out_channels, out_channels * self.expansion, kernel_size=1)

        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PPM(nn.Module):
    def __init__(self, in_channels, ppm_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.out_channels = out_channels

        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, ppm_channels, kernel_size=1),
        )

        self.process = ConvBNReLU(ppm_channels, out_channels, kernel_size=3)
        self.shortcut = ConvBNReLU(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        scale = F.interpolate(
            self.scale(x),
            scale_factor=(int(x.size(2)), int(x.size(3))),
            mode="bilinear",
            align_corners=False,
        )

        process = self.process(scale)
        shortcut = self.shortcut(x)

        return process + shortcut


class Encoder(nn.Module):
    def __init__(self, init_planes=32, ppm_planes=128):
        super().__init__()

        self.planes = init_planes
        self.ppm_planes = ppm_planes

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
            BasicBlock, self.planes * 4, self.planes * 8, 2, 2
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

        self.ppm = PPM(self.planes * 16, self.ppm_planes, self.planes * 4)

        self.out_channels = (self.planes * 4, self.planes * 4, self.planes * 2)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        downsample = None

        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = ConvBNReLU(
                in_channels, out_channels * block.expansion, 1, stride
            )

        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        stem = self.stem(x)

        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)

        context3 = self.context3(layer2)
        detail3 = self.detail3(layer2)

        down3 = context3 + self.down3(detail3)
        compression3 = detail3 + F.interpolate(
            self.compression3(context3),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        context4 = self.context4(down3)
        detail4 = self.detail4(compression3)

        down4 = context4 + self.down4(detail4)
        compression4 = detail4 + F.interpolate(
            self.compression4(context4),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

        context5 = self.context5(down4)
        detail5 = self.detail5(compression4)

        ppm = F.interpolate(
            self.ppm(context5), scale_factor=8, mode="bilinear", align_corners=False
        )

        return (ppm, detail5, compression3)


class Model(nn.Module):
    def __init__(
        self,
        classification_classes=19,
        localization_classes=8,
        encoder_init_planes=32,
        encoder_ppm_planes=128,
        head_planes=64,
    ):
        super().__init__()

        self.classification_classes = classification_classes
        self.localization_classes = localization_classes
        self.encoder_init_planes = encoder_init_planes
        self.encoder_ppm_planes = encoder_ppm_planes
        self.head_planes = head_planes

        self.encoder = Encoder(self.encoder_init_planes, self.encoder_ppm_planes)

        self.classification = nn.Sequential(
            ConvBNReLU(self.encoder.out_channels[0], self.head_planes, 3, 1),
            nn.Conv2d(self.head_planes, self.classification_classes, 1),
        )

        self.localization = nn.Sequential(
            ConvBNReLU(self.encoder.out_channels[1], self.head_planes * 2, 3, 1),
            nn.Conv2d(self.head_planes * 2, 3, 1),
        )

        if self.training:
            self.aux_classification = nn.Sequential(
                ConvBNReLU(self.encoder.out_channels[2], self.head_planes, 3, 1),
                nn.Conv2d(self.head_planes, self.classification_classes, 1),
            )

    def postprocess(
        self,
        classfication,
        localization,
        bbox_down_stride=1,
        conf_th=0.3,
        topK=100,
    ):
        batch_mask = torch.argmax(classfication, dim=1)

        centerness, regression = localization[:, 0, :, :], localization[:, 1:, :, :]

        centerness = centerness.sigmoid()

        center_pool = F.max_pool2d(centerness, kernel_size=3, stride=1, padding=1)
        center_mask = (centerness == center_pool).float()
        centerness = centerness * center_mask

        batch, height, width = centerness.shape

        detections = []

        for idx in range(batch):
            scores, indices = torch.topk(centerness[idx].view(-1), k=topK)
            scores = scores[scores >= conf_th]

            indices = indices[: len(scores)]
            indices = indices % (height * width)

            xs, ys = (indices % width).int(), indices // width

            wh = regression[idx][:, ys, xs]
            half_w, half_h = wh[0] / 2, wh[1] / 2

            bboxes = (
                torch.stack([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=1)
                * bbox_down_stride
            )

            labels = (
                self.classification_classes - self.localization_classes
            ) - batch_mask[idx][ys * bbox_down_stride, xs * bbox_down_stride].int()

            mask = labels >= 0

            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            detections.append((bboxes, scores, labels))

        return batch_mask, detections

    def forward(self, x):
        ppm, detail5, compression3 = self.encoder(x)

        classification = F.interpolate(
            self.classification(ppm + detail5),
            scale_factor=8,
            mode="bilinear",
            align_corners=False,
        )
        localization = self.localization(
            F.interpolate(detail5, scale_factor=2, mode="bilinear", align_corners=False)
        )

        if self.training:
            classfication_aux = F.interpolate(
                self.aux_classification(compression3),
                scale_factor=8,
                mode="bilinear",
                align_corners=False,
            )

            return classification, classfication_aux, localization

        return classification, localization


if __name__ == "__main__":

    model = Model(
        classification_classes=14,
        localization_classes=3,
        encoder_init_planes=32,
        encoder_ppm_planes=128,
        head_planes=64,
    )
    model.eval()
    model.cuda()

    sample = torch.randn(1, 3, 384, 768).cuda()

    classification, localization = model(sample)

    print(classification.shape)
    print(localization.shape)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters (Trainable): {(parameters / 1e6):.2f} M")

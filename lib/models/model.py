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

        self.out_channels = (
            self.planes * 4,
            self.planes * 4,
            self.planes * 2,
            [
                self.planes,
                self.planes * 2,
                self.planes * 4,
                self.planes * 8,
            ],
        )

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

        # ppm = F.interpolate(
        #     self.ppm(context5), scale_factor=8, mode="bilinear", align_corners=False
        # )

        ppm = self.ppm(context5)

        return (
            ppm,
            detail5,
            compression3,
            [layer1, layer2, context3, context4],
        )


class Neck(nn.Module):
    def __init__(self, in_channels, fpn_channels):
        super().__init__()

        (
            self.layer1_channels,
            self.layer2_channels,
            self.context3_channels,
            self.context4_channels,
        ) = in_channels

        self.fpn_channels = fpn_channels

        self.context4 = ConvBNReLU(
            self.context4_channels, self.fpn_channels, kernel_size=3
        )
        self.context3 = ConvBNReLU(
            self.context3_channels, self.fpn_channels, kernel_size=3
        )

        self.layer2 = ConvBNReLU(self.layer2_channels, self.fpn_channels, kernel_size=3)
        self.layer1 = ConvBNReLU(self.layer1_channels, self.fpn_channels, kernel_size=3)

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, ppm, detail5, context4, context3, layer2, layer1):
        ppm = F.interpolate(
            ppm,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        context4 = F.interpolate(
            self.context4(context4) + ppm,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        context3 = F.interpolate(
            self.context3(context3) + context4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        layer2 = F.interpolate(
            self.layer2(layer2) + context3 + detail5,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        layer1 = self.layer1(layer1) + layer2

        return layer1


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, head_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.head_channels = head_channels
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            ConvBNReLU(in_channels, head_channels, kernel_size=3),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),
        )

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.classifier(x)


class LocalizationHead(nn.Module):
    def __init__(self, in_channels, head_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.head_channels = head_channels
        self.num_classes = num_classes

        self.centerness = nn.Sequential(
            ConvBNReLU(self.in_channels, self.head_channels, kernel_size=3),
            nn.Conv2d(self.head_channels, self.num_classes, kernel_size=1),
        )

        self.regression = nn.Sequential(
            ConvBNReLU(self.in_channels, self.head_channels, kernel_size=3),
            nn.Conv2d(self.head_channels, 2, kernel_size=1),
        )

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        centerness = self.centerness(x)
        regression = self.regression(x)

        return centerness, regression


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
        self.neck = Neck(self.encoder.out_channels[-1], self.encoder.out_channels[1])

        self.classification_head = ClassificationHead(
            self.encoder.out_channels[1], self.head_planes, self.classification_classes
        )

        if self.training:
            self.classfication_aux_head = ClassificationHead(
                self.encoder.out_channels[2],
                self.head_planes,
                self.classification_classes,
            )

        self.localization_head = LocalizationHead(
            self.encoder.out_channels[1],
            self.head_planes,
            localization_classes,
        )

    def postprocess(
        self,
        classfication,
        centerness,
        regression,
        bbox_down_stride=1,
        conf_th=0.3,
        topK=100,
    ):
        batch_mask = torch.argmax(classfication, dim=1)

        centerness = centerness.sigmoid()

        center_pool = F.max_pool2d(centerness, kernel_size=3, stride=1, padding=1)
        center_mask = (centerness == center_pool).float()
        centerness = centerness * center_mask

        batch, _, height, width = centerness.shape

        detections = []

        for idx in range(batch):
            scores, indices = torch.topk(centerness[idx].view(-1), k=topK)
            scores = scores[scores >= conf_th]

            indices = indices[: len(scores)]

            labels = indices // (height * width)
            indices = indices % (height * width)

            xs, ys = (indices % width).int(), indices // width

            wh = regression[idx][:, ys, xs]
            half_w, half_h = wh[0] / 2, wh[1] / 2

            bboxes = (
                torch.stack([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=1)
                * bbox_down_stride
            )

            detections.append((bboxes, scores, labels))

        return batch_mask, detections

    def forward(self, x):
        ppm, detail5, compression3, [layer1, layer2, context3, context4] = self.encoder(
            x
        )

        neck = self.neck(ppm, detail5, context4, context3, layer2, layer1)

        classfication = F.interpolate(
            self.classification_head(neck),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        centerness, regression = self.localization_head(neck)

        if self.training:
            classfication_aux = F.interpolate(
                self.classfication_aux_head(compression3),
                scale_factor=8,
                mode="bilinear",
                align_corners=False,
            )

            return classfication, classfication_aux, centerness, regression

        return classfication, centerness, regression


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

    classification, centerness, regression = model(sample)

    print(classification.shape)
    print(centerness.shape)
    print(regression.shape)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters (Trainable): {(parameters / 1e6):.2f} M")

import torch.nn as nn
import torch.nn.functional as F

import timm


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )

        self.feature_channels = self.feature_extractor.feature_info.channels()
        print(self.feature_channels)

        self.c4 = nn.Sequential(
            nn.Conv2d(self.feature_channels[3], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(self.feature_channels[2], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(self.feature_channels[1], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(self.feature_channels[0], 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 19, kernel_size=1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        c4 = F.interpolate(
            self.c4(features[3]), scale_factor=2, mode="bilinear", align_corners=False
        )
        c3 = F.interpolate(
            self.c3(features[2]) + c4,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        c2 = F.interpolate(
            self.c2(features[1]) + c3,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        c1 = self.c1(features[0]) + c2
        out = F.interpolate(
            self.classifier(c1), scale_factor=4, mode="bilinear", align_corners=False
        )
        return out

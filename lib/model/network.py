import torch
from torch import nn
import torch.nn.functional as F

from lib.model.modules import common as cm
from lib.model.backbones.ddrnet import DDRNet
from lib.model.modules.head import ClassificationHead, CenternessHead, RegressionHead


class PixelClassificationModel(nn.Module):

    def __init__(self, classification_classes, head_channels, ppm_block="dappm"):
        super().__init__()

        self.head_channels = head_channels
        self.classification_classes = classification_classes

        self.backbone = DDRNet(ppm_block=ppm_block, planes=32, ppm_planes=128)

        self.classifier = ClassificationHead(
            in_channels=self.backbone.out_channels[0],
            head_channels=head_channels,
            num_classes=classification_classes,
            scale_factor=None,
        )

        cm.init_weights(self)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ppm, detail5, _, aux = self.backbone(input_tensor)

        classifier = self.classifier(ppm + detail5)

        return classifier, aux


class LocalizationModel(nn.Module):

    def __init__(
        self,
        classification_classes,
        localization_classes,
        head_channels,
        backbone_path,
        ppm_block="dappm",
    ):
        super().__init__()

        self.head_channels = head_channels
        self.localization_classes = localization_classes

        self.backbone = PixelClassificationModel(
            classification_classes, head_channels, ppm_block="ce"
        )

        snapshot = torch.load(backbone_path)

        state_dict = snapshot["model_state_dict"]

        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

        self.backbone.load_state_dict(state_dict)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.centerness = CenternessHead(
            in_channels=classification_classes,
            head_channels=head_channels,
            num_classes=localization_classes,
            scale_factor=8,
        )

        self.regressor = RegressionHead(
            in_channels=classification_classes,
            head_channels=head_channels,
            scale_factor=8,
        )

    def forward(self, input_tensor):
        classifier, _ = self.backbone(input_tensor)

        centerness = self.centerness(classifier).sigmoid()
        regression = self.regressor(classifier)

        return classifier, centerness, regression

    def postprocess(self, output, conf_th=0.2):
        _, centerness, regression = output

        center_pool = F.max_pool2d(centerness, kernel_size=3, stride=1, padding=1)
        center_mask = (center_pool == centerness).float()
        centerness = centerness * center_mask

        batch, _, height, width = centerness.shape

        detections = []

        for batch_idx in range(batch):
            scores, indices = torch.topk(centerness[batch_idx].view(-1), k=100)
            scores = scores[scores >= conf_th]
            topk_indices = indices[: len(scores)]

            labels = (topk_indices / (height * width)).int()
            indices = topk_indices % (height * width)

            xs = (indices % width).int()
            ys = (indices / width).int()

            wh = regression[batch_idx][:, ys, xs]
            half_w, half_h = wh[0] / 2, wh[1] / 2

            bboxes = torch.stack(
                [xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=1
            )

            detections.append([bboxes, scores, labels])

        return detections


class Network(nn.Module):
    """
    Network module for pixel-level classification and object localization tasks.

    Args:
        localization_classes (int): Number of localization classes.
        classification_classes (int): Number of classification classes.
        head_channels (int): Number of channels in the head.
        ppm_block (str, optional): Name of the PPM block. Defaults to "dappm".
    """

    def __init__(
        self,
        localization_classes: int,
        classification_classes: int,
        head_channels: int,
        ppm_block: str = "dappm",
    ) -> None:
        """
        Initialize the module.
        """
        super().__init__()

        self.head_channels = head_channels
        self.localization_classes = localization_classes
        self.classification_classes = classification_classes

        # Backbone network
        self.backbone = DDRNet(ppm_block=ppm_block, planes=32, ppm_planes=128)

        # Heads
        self.classifier = ClassificationHead(
            in_channels=self.backbone.out_channels[0],
            head_channels=head_channels,
            num_classes=classification_classes,
            scale_factor=None,
        )

        self.centerness = CenternessHead(
            in_channels=localization_classes,
            head_channels=head_channels,
            num_classes=localization_classes,
        )

        self.regressor = RegressionHead(
            in_channels=localization_classes,
            head_channels=head_channels,
        )

        # Initialize weights
        cm.init_weights(self)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of output pixel-level classification
            and object localization tensors.
        """
        # Pass input through backbone network
        ppm, detail5, _, aux = self.backbone(input_tensor)

        classifier = self.classifier(ppm + detail5)
        centerness = self.centerness(classifier[:, 11::, :]).sigmoid()
        regression = self.regressor(classifier[:, 11::, :])

        classifier = cm.Upsample(classifier, scale_factor=8)

        # Pass output through classification and localization heads
        return (classifier, aux), (centerness, regression)


if __name__ == "__main__":
    model = Network(
        localization_classes=7,
        classification_classes=19,
        head_channels=32,
        ppm_block="dappm",
    )
    model.eval()
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.randn(1, 3, 1024, 1024).to(device)
    out = model(x)

    print(
        f"Classification output shape: {out[0].shape}, Localization output shape: (Centerness: {out[1][0].shape}, Regression: {out[1][1].shape})"
    )

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parameters / 1e6:.2f}M")  # 3.87M

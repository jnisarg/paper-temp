import torch
from torch import nn

from lib.model.modules import common as cm
from lib.model.backbones.ddrnet import DDRNet
from lib.model.modules.head import ClassificationHead, CenternessHead, RegressionHead


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
        )

        self.centerness = CenternessHead(
            in_channels=classification_classes,
            head_channels=head_channels,
            num_classes=localization_classes,
        )

        self.regressor = RegressionHead(
            in_channels=self.backbone.out_channels[0],
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
        ppm, detail5, _ = self.backbone(input_tensor)

        classifier = self.classifier(ppm + detail5)
        centerness = self.centerness(classifier)
        regression = self.regressor(ppm + detail5)

        # Pass output through classification and localization heads
        return classifier, (centerness, regression)


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

import torch
import torch.nn as nn

from lib.model.network import Network
from lib.utils.criterion import Criterion


class FullModel(nn.Module):

    def __init__(self, model, criterion):
        super().__init__()

        self.model = model
        self.criterion = criterion

    def _pixel_accuracy(self, pred, target):
        keep = target != self.criterion.ignore_index

        target = target[keep]
        pred = torch.argmax(pred, dim=1)[keep]

        return torch.eq(target, pred).sum().float() / keep.sum().float()

    def forward(self, targets):
        images, bboxes, labels, heatmaps, masks, infos = targets
        preds = self.model(images)

        pixel_accuracy = self._pixel_accuracy(preds[0], masks)
        classification_loss, localization_loss, centerness_loss, bbox_loss = (
            self.criterion(preds, targets)
        )

        return (
            preds,
            pixel_accuracy,
            classification_loss,
            localization_loss,
            centerness_loss,
            bbox_loss,
        )


def build_model():
    model = Network(8, 19, head_channels=64, ppm_block="ce")
    criterion = Criterion()

    return FullModel(model, criterion)

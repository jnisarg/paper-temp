import torch
import torch.nn as nn

from lib.model.network import PixelClassificationModel, LocalizationModel
from lib.utils.criterion import PixelClassificationCriterion, LocalizationCriterion


class Model(nn.Module):

    def __init__(self, model, criterion):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.postprocess = self.model.postprocess if isinstance(self.model, LocalizationModel) else None

    def forward(self, targets):

        preds = self.model(targets[0])
        loss = self.criterion(preds, targets)

        return preds, loss


def build_pixel_classification_model():
    return Model(
        PixelClassificationModel(19, head_channels=64, ppm_block="ce"),
        PixelClassificationCriterion(),
    )


def build_localization_model(backbone_path):
    return Model(
        LocalizationModel(19, 8, head_channels=64, backbone_path=backbone_path),
        LocalizationCriterion(),
    )


# class FullModel(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.model = Network(8, 19, head_channels=64, ppm_block="ce")
#         self.pixel_classification_criterion = PixelClassificationCriterion()
#         self.localization_criterion = LocalizationCriterion()

#         self.epoch = 0

#     def _pixel_accuracy(self, pred, target):
#         keep = target != self.pixel_classification_criterion.ignore_index

#         target = target[keep]
#         pred = torch.argmax(pred, dim=1)[keep]

#         return torch.eq(target, pred).sum().float() / keep.sum().float()

#     def update(self, epoch):
#         self.epoch = epoch

#     def forward(self, targets):

#         preds = self.model(targets[0])
#         pixel_accuracy = self._pixel_accuracy(preds[0][0], targets[1])

#         classification_loss = self.pixel_classification_criterion(preds[0], targets[1])

#         if self.epoch >= 1:
#             localization_loss, (centerness_loss, bbox_loss) = (
#                 self.localization_criterion(preds[1], targets)
#             )

#             return (
#                 preds,
#                 classification_loss,
#                 localization_loss,
#                 (centerness_loss, bbox_loss),
#                 pixel_accuracy,
#             )

#         return preds, classification_loss, pixel_accuracy

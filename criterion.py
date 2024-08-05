import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.thresh = -torch.log(torch.tensor(0.8, dtype=torch.float))
        self.ignore_index = 255

    def _ohem_loss(self, pred, target):
        n_min = target[target != self.ignore_index].numel() // 16

        loss = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction="none"
        ).view(-1)

        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return loss_hard.mean()

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive_indices = target.eq(1).float()
        negative_indices = target.lt(1).float()

        negative_weights = torch.pow(1 - target, 4)
        pred = torch.clamp(pred, torch.finfo(pred.dtype).tiny)

        positive_loss = torch.log(pred) * torch.pow(1 - pred, 2) * positive_indices
        negative_loss = torch.log(1 - pred + torch.finfo(pred.dtype).tiny)
        negative_loss = (
            negative_loss * torch.pow(pred, 2) * negative_weights * negative_indices
        )

        num_positive = torch.sum(positive_indices)
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()
        # print(
        #     positive_loss.item(),
        #     negative_loss.item(),
        #     (1 - pred + torch.finfo(pred.dtype).tiny).min().item(),
        #     (1 - pred + torch.finfo(pred.dtype).tiny).max().item(),
        #     pred.min().item(),
        #     pred.max().item(),
        #     num_positive.item(),
        # )

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss

    def forward(self, preds, targets):
        classifier, compression3, centerness, regression = preds
        images, masks, bboxes, labels, heatmaps, infos = targets

        classifier_loss = self._ohem_loss(classifier, masks) + 0.4 * self._ohem_loss(
            compression3, masks
        )

        target_nonpad_mask = labels.gt(-1)

        centerness_loss = self._focal_loss(centerness, heatmaps)
        regression_loss = centerness_loss.new_tensor(0.0)

        num = 0
        for batch in range(images.size(0)):
            ct = infos[batch]["bbox_centers"].cuda()
            ct_int = ct.long()
            num += len(ct_int)

            batch_regression_pred = regression[
                batch, :, ct_int[:, 1], ct_int[:, 0]
            ].view(-1)
            batch_bboxes = bboxes[batch][target_nonpad_mask[batch]]

            wh = (
                torch.stack(
                    [
                        batch_bboxes[:, 2] - batch_bboxes[:, 0],
                        batch_bboxes[:, 3] - batch_bboxes[:, 1],
                    ]
                ).view(-1)
                / 4
            )

            regression_loss += F.smooth_l1_loss(
                batch_regression_pred, wh, reduction="sum"
            )

        bbox_loss = regression_loss / (num + 1e-7) * 0.1
        localization_loss = centerness_loss + bbox_loss

        return classifier_loss + localization_loss
        # return classifier_loss

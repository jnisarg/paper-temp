import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(
        self,
        ohem_thresh=0.8,
        n_min_divisor=16,
        alpha=2,
        gamma=4,
        box_loss="l1_loss",
        centerness_weight=1.0,
        regression_weight=0.7,
        localization_weight=1.0,
        classification_weight=1.0,
        eps=1e-7,
        # down_stride=4,
        ignore_index=255,
    ):
        super().__init__()

        assert box_loss in ["l1_loss", "smooth_l1_loss"]

        self.ohem_thresh = -torch.log(torch.tensor(ohem_thresh, dtype=torch.float))
        self.n_min_divisor = n_min_divisor
        self.alpha = alpha
        self.gamma = gamma
        self.box_loss = eval("F." + box_loss)
        self.centerness_weight = centerness_weight
        self.regression_weight = regression_weight
        self.localization_weight = localization_weight
        self.classification_weight = classification_weight
        self.eps = eps
        # self.down_stride = down_stride
        self.ignore_index = ignore_index

    def _ohem_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_min = target[target != self.ignore_index].numel() // self.n_min_divisor

        loss = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction="none"
        ).view(-1)

        loss_hard = loss[loss > self.ohem_thresh]

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
        # print(negative_loss.sum().item())
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
        #     num_positive.item(),
        # )

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss

    def forward(self, pred, target):
        images, bboxes, labels, heatmaps, masks, infos = target

        target_nonpad_mask = labels.gt(-1)

        centerness_loss = self._focal_loss(pred[1][0], heatmaps)
        regression_loss = centerness_loss.new_tensor(0.0)

        classification_loss = (
            self._ohem_loss(pred[0], masks) * self.classification_weight
        )

        num = 0
        for batch in range(images.size(0)):
            ct = infos[batch]["ct"].cuda()
            ct_int = ct.long()
            num += len(ct_int)

            batch_regression_pred = pred[1][1][
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
                # / self.down_stride
            )

            regression_loss += (
                self.box_loss(batch_regression_pred, wh, reduction="sum")
                * self.regression_weight
            )

        # print(pred[1][0], heatmaps)
        # exit()

        bbox_loss = regression_loss / (num + self.eps) * 0.1

        localization_loss = (centerness_loss + bbox_loss) * self.localization_weight

        return classification_loss, localization_loss, centerness_loss, bbox_loss

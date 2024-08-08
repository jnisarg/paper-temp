import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(
        self,
        ohem_ratio=0.7,
        n_min_divisor=16,
        ignore_index=255,
        focal_alpha=0.75,
        focal_gamma=2,
        bbox_loss="smooth_l1",
        classification_loss_weight=1.0,
        centerness_loss_weight=1.0,
        bbox_loss_weight=0.1,
        localization_loss_weight=1.0,
    ):
        super().__init__()

        assert bbox_loss in ["smooth_l1", "l1"]

        self.ohem_ratio = ohem_ratio
        self.n_min_divisor = n_min_divisor
        self.ignore_index = ignore_index
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bbox_loss = F.l1_loss if bbox_loss == "l1" else F.smooth_l1_loss

        self.classification_loss_weight = classification_loss_weight
        self.centerness_loss_weight = centerness_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.localization_loss_weight = localization_loss_weight

    def _ohem_loss(self, pred, target):
        # Compute the cross-entropy loss for each pixel
        loss = F.cross_entropy(
            pred, target, reduction="none", ignore_index=self.ignore_index
        )

        # Flatten the loss and target
        loss = loss.view(-1)
        target = target.view(-1)

        # Filter out the ignored target
        valid_mask = target != self.ignore_index
        loss = loss[valid_mask]

        # Sort the loss values in descending order
        sorted_loss, _ = torch.sort(loss, descending=True)

        # Select the top ratio of hard examples
        num_hard_examples = int(self.ohem_ratio * sorted_loss.size(0))
        hard_loss = sorted_loss[:num_hard_examples]

        # Return the mean of the hard examples
        return hard_loss.mean()

    def _focal_loss(self, pred, target):
        # Compute the binary cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Compute the probability of the positive class
        pt = torch.exp(-BCE_loss)

        # Compute the focal loss
        F_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * BCE_loss

        return F_loss.mean()

    def forward(self, outputs, targets):
        # if len(outputs) == 3:
        #     classification, centerness, regression = outputs
        # elif len(outputs) == 4:
        #     classification, aux_classification, centerness, regression = outputs
        # else:
        #     raise NotImplementedError("The number of targets should be either 3 or 4.")

        classification, centerness, regression = outputs

        images, masks, bboxes, labels, bbox_center_heatmaps, infos = targets

        # if len(outputs) == 4:
        #     classification_loss = (
        #         self._ohem_loss(classification, masks)
        #         + self._ohem_loss(aux_classification, masks)
        #     ) * self.classification_loss_weight
        # else:
        #     classification_loss = (
        #         self._ohem_loss(classification, masks) * self.classification_loss_weight
        #     )

        classification_loss = (
            self._ohem_loss(classification, masks) * self.classification_loss_weight
        )

        centerness_loss = (
            self._focal_loss(centerness, bbox_center_heatmaps)
            * self.centerness_loss_weight
        )
        regression_loss = centerness_loss.new_tensor(0.0)

        target_nonpad_mask = labels.gt(-1)

        num = 0
        for idx in range(len(images)):
            ct = infos[idx]["bbox_centers"].cuda()
            ct_int = ct.long()
            num += len(ct_int)

            batch_regression = regression[idx, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_bboxes = bboxes[idx][target_nonpad_mask[idx]]

            wh = (
                torch.stack(
                    [
                        batch_bboxes[:, 2] - batch_bboxes[:, 0],
                        batch_bboxes[:, 3] - batch_bboxes[:, 1],
                    ]
                )
                / infos[idx]["bbox_down_stride"]
            ).view(-1)

            regression_loss += self.bbox_loss(batch_regression, wh, reduction="sum")

        bbox_loss = regression_loss / (num + 1e-7) * self.bbox_loss_weight
        localization_loss = (
            centerness_loss + bbox_loss
        ) * self.localization_loss_weight

        return classification_loss + localization_loss, (
            classification_loss,
            centerness_loss,
            bbox_loss,
        )

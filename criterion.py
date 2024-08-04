import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

        self.thresh = -torch.log(torch.tensor(0.75, dtype=torch.float))
        self.ignore_index = 255

    def forward(self, preds, masks):
        n_min = masks[masks != self.ignore_index].numel() // 16

        loss = F.cross_entropy(
            preds, masks, ignore_index=self.ignore_index, reduction="none"
        ).view(-1)

        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return loss_hard.mean()

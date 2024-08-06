from timm.optim import AdaBelief

# from timm.scheduler import PolyLRScheduler
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, PolynomialLR

import lightning as L
from torchmetrics.detection import MeanAveragePrecision

from model import Model

from metrics import Metrics
from criterion import Criterion
from cityscapes import CityscapesDataModule


def freeze_od(model):
    for param in model.centerness.parameters():
        param.requires_grad = False

    for param in model.regression.parameters():
        param.requires_grad = False


def unfreeze_od(model):
    for param in model.centerness.parameters():
        param.requires_grad = True

    for param in model.regression.parameters():
        param.requires_grad = True


def freeze_sd(model):
    for param in model.classifier.parameters():
        param.requires_grad = False

    for param in model.compression3.parameters():
        param.requires_grad = False


def unfreeze_sd(model):
    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.compression3.parameters():
        param.requires_grad = True


class Network(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model(enc_planes=48, enc_ppm_planes=128, head_planes=256)
        self.criterion = Criterion()

        self.class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            # "rider",
            "car",
            # "truck",
            # "bus",
            # "train",
            # "motorcycle",
            "bicycle",
        ]
        self.metrics = Metrics(num_classes=14, class_names=self.class_names)

        self.mAP = MeanAveragePrecision(class_metrics=True, iou_thresholds=[0.5]).cuda(
            device=1
        )

    def training_step(self, batch):

        if self.trainer.global_step > 30_000 and self.trainer.global_step <= 150_000:
            freeze_od(self.model)
        elif self.trainer.global_step > 150_000 and self.trainer.global_step <= 270_000:
            unfreeze_od(self.model)
            freeze_sd(self.model)
        elif self.trainer.global_step > 270_000:
            unfreeze_sd(self.model)

        preds = self.model(batch[0])
        loss = self.criterion(preds, batch, self.trainer.global_step)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            # batch_size=16,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            # batch_size=16,
        )

        return loss

    def validation_step(self, batch):
        preds = self.model(batch[0])
        loss = self.criterion(preds, batch)

        self.metrics.update(preds[0], batch[1])

        _, detections = self.postprocess(preds, conf_th=0.3, topk=100)

        pred_det = [
            {
                "boxes": detections[0][0],
                "scores": detections[0][1],
                "labels": detections[0][2],
            }
        ]

        target_det = [{"boxes": batch[2][0], "labels": batch[3][0]}]

        self.mAP.update(pred_det, target_det)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        return loss

    def on_validation_epoch_end(self):
        self.metrics.collect()
        mAP = self.mAP.compute()

        self.log(
            "val_miou",
            self.metrics.metrics["iou"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        self.log(
            "val_mAP",
            mAP["map_50"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=1,
        )

        self.print(f"\n\n{self.metrics}\n{mAP}\n\n")

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.045, weight_decay=0.0005)

        if self.trainer.global_step > 150_000:
            lr = 0.0001
        else:
            lr = 0.001

        optimizer = AdaBelief(self.parameters(), lr=lr, weight_decay=0.0005)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10_000, T_mult=1, eta_min=1e-6
        )

        # scheduler = PolyLRScheduler(optimizer, t_initial=100, lr_min=1e-6, power=0.9)
        # scheduler = PolynomialLR(optimizer, total_iters=240_000, power=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                # "frequency": 1,
            },
        }

    def postprocess(self, outputs, conf_th=0.3, topk=100):
        classifier, _, centerness, regression = outputs

        center_pool = F.max_pool2d(centerness, kernel_size=3, stride=1, padding=1)
        center_mask = (center_pool == centerness).float()
        centerness = centerness * center_mask

        batch, _, height, width = centerness.shape

        detections = []

        for batch_idx in range(batch):
            scores, indices = torch.topk(centerness[batch_idx].view(-1), k=topk)
            scores = scores[scores >= conf_th]
            topk_indices = indices[: len(scores)]

            labels = (topk_indices / (height * width)).int()
            indices = topk_indices % (height * width)

            xs = (indices % width).int()
            ys = (indices / width).int()

            wh = regression[batch_idx][:, ys, xs]
            half_w, half_h = wh[0] / 2, wh[1] / 2

            bboxes = (
                torch.stack([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=1)
                * 4
            )

            detections.append([bboxes, scores, labels])

        return classifier.argmax(dim=1), detections

    def forward(self, x):
        return self.model(x)


def main():
    dm = CityscapesDataModule()
    dm.setup()

    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir="logs", name="cityscapes_bs8_b48_h256_freeze"
    )

    model = Network()
    trainer = L.Trainer(
        accelerator="gpu",
        # strategy="ddp",
        devices=[1],
        logger=logger,
        precision="16-mixed",
        # max_epochs=100,
        max_steps=330_000,
        # callbacks=[
        #     L.pytorch.callbacks.early_stopping.EarlyStopping(
        #         monitor="val_loss", mode="min"
        #     )
        # ],
        # benchmark=True,
        enable_checkpointing=True,
        enable_model_summary=True,
        check_val_every_n_epoch=5,
        # log_every_n_steps=10,
        sync_batchnorm=True,
        default_root_dir="checkpoints",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

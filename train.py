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


class Network(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
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
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.metrics = Metrics(num_classes=19, class_names=self.class_names)

        self.mAP = MeanAveragePrecision()

    def training_step(self, batch):
        preds = self.model(batch[0])
        loss = self.criterion(preds, batch)

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

        self.print(f"\n\n{self.metrics}\n{mAP}\n\n")

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.045, weight_decay=0.0005)
        optimizer = AdaBelief(self.parameters(), lr=0.001, weight_decay=0.0005)

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10_000, T_mult=1, eta_min=1e-6
        )

        # scheduler = PolyLRScheduler(optimizer, t_initial=100, lr_min=1e-6, power=0.9)
        # scheduler = PolynomialLR(optimizer, total_iters=120_000, power=0.9)

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

            bboxes = torch.stack(
                [xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=1
            )

            detections.append([bboxes, scores, labels])

        return classifier.argmax(dim=1), detections

    def forward(self, x):
        return self.model(x)


def main():
    dm = CityscapesDataModule()
    dm.setup()

    logger = L.pytorch.loggers.TensorBoardLogger(save_dir="logs", name="cityscapes")

    model = Network()
    trainer = L.Trainer(
        accelerator="gpu",
        # strategy="ddp",
        devices=[1],
        logger=logger,
        precision="16-mixed",
        # max_epochs=100,
        max_steps=120_000,
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
        sync_batchnorm=False,
        default_root_dir="checkpoints",
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

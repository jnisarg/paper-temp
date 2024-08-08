import lightning as L
import torch
from torchmetrics.detection import MeanAveragePrecision

from timm.optim import AdaBelief

from lib.criterion import Criterion
from lib.metrics import Metrics
from lib.models import Model


class MTL(L.LightningModule):

    def __init__(
        self,
        classification_class_names,
        localization_class_names,
        encoder_init_planes=32,
        encoder_ppm_planes=128,
        head_planes=64,
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
        eps=1e-6,
        conf_th=0.3,
        topK=100,
        lr=1e-3,
        weight_decay=5e-4,
    ):
        super().__init__()

        self.example_input_array = torch.rand(1, 3, 384, 768)

        self.classification_class_names = classification_class_names
        self.localization_class_names = localization_class_names

        self.encoder_init_planes = encoder_init_planes
        self.encoder_ppm_planes = encoder_ppm_planes
        self.head_planes = head_planes

        self.ohem_ratio = ohem_ratio
        self.n_min_divisor = n_min_divisor
        self.ignore_index = ignore_index
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.bbox_loss = bbox_loss

        self.classification_loss_weight = classification_loss_weight
        self.centerness_loss_weight = centerness_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.localization_loss_weight = localization_loss_weight

        self.eps = eps

        self.conf_th = conf_th
        self.topK = topK

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = Model(
            classification_classes=len(classification_class_names),
            localization_classes=len(localization_class_names),
            encoder_init_planes=encoder_init_planes,
            encoder_ppm_planes=encoder_ppm_planes,
            head_planes=head_planes,
        )
        self.criterion = Criterion(
            ohem_ratio=ohem_ratio,
            n_min_divisor=n_min_divisor,
            ignore_index=ignore_index,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            bbox_loss=bbox_loss,
            classification_loss_weight=classification_loss_weight,
            centerness_loss_weight=centerness_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            localization_loss_weight=localization_loss_weight,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch[0])
        loss, (
            classification_loss,
            centerness_loss,
            bbox_loss,
        ) = self.criterion(outputs, batch)

        self.log_dict(
            {
                "train_classification_loss": classification_loss,
                "train_centerness_loss": centerness_loss,
                "train_bbox_loss": bbox_loss,
                "train_loss": loss,
                "lr": self.optimizers().param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_validation_epoch_start(self):
        self.iou_metrics = Metrics(
            class_names=self.classification_class_names,
            table_fmt="fancy_grid",
            missing_val="-",
            ignore_index=self.ignore_index,
            eps=self.eps,
        )

        self.ap_metrics = MeanAveragePrecision(
            class_metrics=True, iou_thresholds=[0.5]
        ).cuda()

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch[0])
        loss, _ = self.criterion(outputs, batch)

        _, detections = self.model.postprocess(
            *outputs,
            bbox_down_stride=batch[-1][0]["bbox_down_stride"],
            conf_th=self.conf_th,
            topK=self.topK,
        )

        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.iou_metrics.update(outputs[0], batch[1])
        self.ap_metrics.update(
            [
                {
                    "boxes": detections[0][0],
                    "scores": detections[0][1],
                    "labels": detections[0][2],
                }
            ],
            [
                {
                    "boxes": batch[2][0],
                    "labels": batch[3][0],
                }
            ],
        )

    def on_validation_epoch_end(self):
        self.iou_metrics.collect()
        mAP = self.ap_metrics.compute()

        self.log_dict(
            {
                "map_50": mAP["map_50"],
                "mIoU": self.iou_metrics.metrics["iou"].mean(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.print(f"\n\n{self.iou_metrics}\n{mAP}\n\n")

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = AdaBelief(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.trainer.max_steps, power=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

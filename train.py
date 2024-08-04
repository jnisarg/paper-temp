from typing import Any
from timm.optim import AdaBelief
from timm.scheduler import PolyLRScheduler

import lightning as L

from model import Model
from metrics import Metrics
from criterion import Criterion
from cityscapes import CityscapesDataModule


class Network(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.criterion = Criterion()

    def training_step(self, batch):
        images, masks = batch

        preds = self.model(images)
        loss = self.criterion(preds, masks)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        images, masks = batch

        self.metrics = Metrics(num_classes=19)

        preds = self.model(images)
        loss = self.criterion(preds, masks)

        self.metrics.update(preds, masks)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_end(self):
        self.metrics.collect()
        self.log(
            "val_miou",
            self.metrics.metrics["iou"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.045, weight_decay=0.0005)
        optimizer = AdaBelief(self.parameters(), lr=0.045, weight_decay=0.0005)

        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=2, eta_min=1e-6
        # )

        scheduler = PolyLRScheduler(optimizer, t_initial=100, lr_min=1e-6, power=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    dm = CityscapesDataModule()
    dm.setup()

    logger = L.pytorch.loggers.TensorBoardLogger(save_dir="logs", name="cityscapes")

    model = Network()
    trainer = L.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=4,
        logger=logger,
        precision="16-mixed",
        max_epochs=100,
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
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

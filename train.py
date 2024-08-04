import timm
import lightning as L

from model import Model
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
        loss, aux_loss = self.criterion(preds, masks)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_aux_loss",
            aux_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch):
        images, masks = batch

        preds = self.model(images)
        loss, aux_loss = self.criterion(preds, masks)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_aux_loss",
            aux_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.045, weight_decay=0.0005)
        optimizer = timm.optim.AdaBelief(
            self.parameters(), lr=0.045, weight_decay=0.0005
        )
        scheduler = timm.scheduler.CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=1e-6,
            decay_rate=0.75,
            warmup_t=2,
            warmup_lr_init=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
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
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

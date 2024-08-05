from timm.optim import AdaBelief

# from timm.scheduler import PolyLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, PolynomialLR

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

        self.print(f"\n\n{self.metrics}\n\n")

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


def main():
    dm = CityscapesDataModule()
    dm.setup()

    logger = L.pytorch.loggers.TensorBoardLogger(save_dir="logs", name="cityscapes")

    model = Network()
    model.to_onnx("model.onnx", input_sample=dm.train_dataloader().dataset[0][0])
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
    model.to_onnx("model_final.onnx", input_sample=dm.train_dataloader().dataset[0][0])

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

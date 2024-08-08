import os

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import callbacks as pl_callbacks

from lib.core import MTL
from lib.data import CityscapesDataModule


def train(
    exp_name,
    save_dir="exp",
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
    root="data/cityscapes",
    train_size=(384, 768),
    test_size=(384, 768),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    bbox_down_stride=4,
    bbox_format="pascal_voc",
    transforms_kwargs={},
    eval_transforms_kwargs={},
    train_batch_size=8,
    test_batch_size=1,
    num_workers=8,
    precision="16-mixed",
    max_steps=120_000,
):

    dm = CityscapesDataModule(
        root=root,
        train_size=train_size,
        test_size=test_size,
        mean=mean,
        std=std,
        ignore_index=ignore_index,
        bbox_down_stride=bbox_down_stride,
        bbox_format=bbox_format,
        transforms_kwargs=transforms_kwargs,
        eval_transforms_kwargs=eval_transforms_kwargs,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
    )
    dm.setup("fit")

    model = MTL(
        classification_class_names=dm.train_dataset.classification_class_names,
        localization_class_names=dm.train_dataset.localization_class_names,
        encoder_init_planes=encoder_init_planes,
        encoder_ppm_planes=encoder_ppm_planes,
        head_planes=head_planes,
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
        eps=eps,
        conf_th=conf_th,
        topK=topK,
        lr=lr,
        weight_decay=weight_decay,
    )

    work_dir = os.path.join(save_dir, exp_name)

    tb_logger = pl_loggers.TensorBoardLogger(work_dir, name="tb_logs", log_graph=True)
    csv_logger = pl_loggers.CSVLogger(work_dir, name="csv_logs")

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=work_dir,
        filename="checkpoint-{epoch:02d}-{mIoU:.2f}-{map_50:.2f}",
        monitor="mIoU",
        mode="max",
        save_last=True,
        save_top_k=5,
        every_n_epochs=5,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        callbacks=[checkpoint_callback],
        logger=[tb_logger, csv_logger],
        max_steps=max_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=5,
        benchmark=True,
        default_root_dir=work_dir,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train(exp_name="cityscapes_default_exp0")

import os
import datetime

import time
import warnings

import torch
import torch.cuda.amp as amp

from lib.core.functions import (
    save_snapshot,
    train_one_epoch_pixel_classification,
    train_one_epoch_localization,
    validate_pixel_classification,
    validate_localization,
)
from lib.data.cityscapes import (
    CityscapesPixelClassificationDataset,
    CityscapesLocalizationDataset,
)
from lib.data.utils import build_dataloader
from lib.model.utils import build_pixel_classification_model, build_localization_model
from lib.utils.lr_scheduler import WarmupPolyLR
from lib.utils.utils import SmoothedValue, initialize_logger, set_seed

warnings.filterwarnings("ignore")


def main():
    set_seed(12467)

    work_dir, logger, writer = initialize_logger(
        work_dir="exp",
        exp_name="paper_exp2_0.1_bbox_loss",
        use_tb=True,
        tb_name="tb_logs",
    )

    pixel_classification_work_dir = os.path.join(work_dir, "pixel_classification")
    localization_work_dir = os.path.join(work_dir, "localization")

    os.makedirs(pixel_classification_work_dir, exist_ok=True)
    os.makedirs(localization_work_dir, exist_ok=True)

    train_start = time.time()

    train_pixel_classification(pixel_classification_work_dir, logger, writer)
    train_localization(
        localization_work_dir,
        logger,
        writer,
        backbone_path=os.path.join(
            pixel_classification_work_dir,
            "checkpoints",
            "model_best.pth",
            # "exp/paper_exp1_0.1_bbox_loss/pixel_classification/checkpoints/model_best.pth"
        ),
    )

    train_end = time.time()
    logger.info(
        f"Training completed in {datetime.timedelta(seconds=int(train_end - train_start))} ... :)"
    )

    return work_dir, logger, writer


def train_pixel_classification(work_dir, logger, writer):
    train_dl, test_dl = build_dataloader(
        train=True, dataset=CityscapesPixelClassificationDataset, logger=logger
    )

    model = build_pixel_classification_model()

    amp_scaler = amp.GradScaler(enabled=True)

    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.045,
        weight_decay=0.0005,
    )

    epochs = 100

    lr_scheduler = WarmupPolyLR(
        optimizer,
        power=0.9,
        max_iter=(epochs * len(train_dl)),
        warmup_iter=0,
    )

    best_score = 0.0

    logger.info("Initializing training (Pixel Classification) ...\n")

    end = time.time()

    epoch_time = SmoothedValue(fmt="{avg:.4f}")

    for epoch in range(epochs):

        train_stats = train_one_epoch_pixel_classification(
            epoch,
            model,
            train_dl,
            optimizer,
            lr_scheduler,
            amp_scaler,
            logger,
            work_dir,
            amp_enabled=True,
            writer=writer,
            log_interval=100,
        )

        if epoch % 5 == 0 or epoch > epochs - 10:
            val_stats = validate_pixel_classification(
                epoch, model, test_dl, logger, writer
            )

            if val_stats["miou"] >= best_score:
                best_score = val_stats["miou"]
                save_snapshot(
                    epoch,
                    model,
                    optimizer,
                    val_stats,
                    work_dir,
                    logger=logger,
                    is_final=False,
                    is_best=True,
                )

            logger.info(f"Val: {val_stats}\n")

        epoch_time.update(time.time() - end)
        end = time.time()

        logger.info(
            f"Epoch {epoch}\n"
            f"Train: {train_stats}\n"
            f"Epoch time: {datetime.timedelta(seconds=int(epoch_time.avg))}\n"
            f"ETA: {datetime.timedelta(seconds=int(epoch_time.avg * (epochs - epoch - 1)))}\n"
        )

    save_snapshot(
        epoch,
        model,
        optimizer,
        val_stats,
        work_dir,
        logger=logger,
        is_final=True,
        is_best=False,
    )


def train_localization(work_dir, logger, writer, backbone_path=None):
    train_dl, test_dl = build_dataloader(
        train=True, dataset=CityscapesLocalizationDataset, logger=logger
    )

    model = build_localization_model(backbone_path=backbone_path)

    amp_scaler = amp.GradScaler(enabled=True)

    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0005,
    )

    epochs = 50

    lr_scheduler = WarmupPolyLR(
        optimizer,
        power=0.9,
        max_iter=(epochs * len(train_dl)),
    )

    best_score = 0.0

    logger.info("\n\nInitializing training (Localization) ...\n")

    end = time.time()

    epoch_time = SmoothedValue(fmt="{avg:.4f}")

    for epoch in range(epochs):

        train_stats = train_one_epoch_localization(
            epoch,
            model,
            train_dl,
            optimizer,
            lr_scheduler,
            amp_scaler,
            logger,
            work_dir,
            amp_enabled=True,
            writer=writer,
            log_interval=100,
        )

        if epoch % 5 == 0 or epoch > epochs - 10:
            val_stats = validate_localization(epoch, model, test_dl, logger, writer)

            if val_stats["mAP"] >= best_score:
                best_score = val_stats["mAP"]
                save_snapshot(
                    epoch,
                    model,
                    optimizer,
                    val_stats,
                    work_dir,
                    logger=logger,
                    is_final=False,
                    is_best=True,
                )

            logger.info(f"Val: {val_stats}\n")

        epoch_time.update(time.time() - end)
        end = time.time()

        logger.info(
            f"Epoch {epoch}\n"
            f"Train: {train_stats}\n"
            f"Epoch time: {datetime.timedelta(seconds=int(epoch_time.avg))}\n"
            f"ETA: {datetime.timedelta(seconds=int(epoch_time.avg * (epochs - epoch - 1)))}\n"
        )

    save_snapshot(
        epoch,
        model,
        optimizer,
        val_stats,
        work_dir,
        logger=logger,
        is_final=True,
        is_best=False,
    )


if __name__ == "__main__":
    # train(config=load_config())
    main()

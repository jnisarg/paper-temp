import datetime

# import os
import time
import warnings

import torch
import torch.cuda.amp as amp

import torch.nn as nn

# import yaml

from lib.core.functions import save_snapshot, train_one_epoch, validate
from lib.data.utils import build_dataloader
from lib.model.utils import build_model
from lib.utils.lr_scheduler import WarmupPolyLR
from lib.utils.utils import SmoothedValue, initialize_logger, set_seed

warnings.filterwarnings("ignore")


def train():
    set_seed(12467)

    work_dir, logger, writer = initialize_logger(
        work_dir="exp",
        exp_name="paper_exp0",
        use_tb=True,
        tb_name="tb_logs",
    )

    train_dl, test_dl = build_dataloader(train=True, logger=logger)

    model = build_model()
    amp_scaler = amp.GradScaler(enabled=True)

    # if len(config["gpus"]) > 1:
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # if config["train"]["sync_bn"]:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    # optimizer = torch.optim.AdamW(  # AdamW is more generalizable than Adam
    #     model.parameters(),
    #     lr=config["train"]["lr"],
    #     weight_decay=config["train"]["weight_decay"],
    # )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0005,
    )

    lr_scheduler = WarmupPolyLR(
        optimizer,
        power=0.9,
        max_iter=(484 * len(train_dl)),
    )

    # with open(os.path.join(work_dir, "config.yml"), "w") as fw:
    #     yaml.dump(config, fw)

    best_score = 0.0

    logger.info("Initializing training ...\n")

    train_start = time.time()
    end = time.time()

    epoch_time = SmoothedValue(fmt="{avg:.4f}")

    for epoch in range(484):
        # val_stats = validate(epoch, model, test_dl, logger, writer)

        train_stats = train_one_epoch(
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

        if epoch % 5 == 0 or epoch > 484 - 10:
            val_stats = validate(epoch, model, test_dl, logger, writer)

            if val_stats["miou"] + val_stats["mAP"] >= best_score:
                best_score = val_stats["miou"]+ val_stats["mAP"]
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
            f"ETA: {datetime.timedelta(seconds=int(epoch_time.avg * (484 - epoch - 1)))}\n"
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

    train_end = time.time()
    logger.info(
        f"Training completed in {datetime.timedelta(seconds=int(train_end - train_start))} ... :)"
    )


if __name__ == "__main__":
    # train(config=load_config())
    train()

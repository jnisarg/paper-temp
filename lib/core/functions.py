import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

from torchmetrics.detection.mean_ap import MeanAveragePrecision

plt.style.use("dark_background")


from lib.utils.metrics import Metrics
from lib.utils.utils import SmoothedValue


def check_if_wrapped(model):
    return isinstance(model, (DataParallel, DistributedDataParallel))


def get_figure():
    pass


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda param: param.grad is not None, parameters))

    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), float(norm_type)).to(device)
                for p in parameters
            ]
        ),
        norm_type,
    )

    return total_norm


def save_snapshot(
    epoch, model, optimizer, stats, work_dir, logger, is_best=False, is_final=False
):
    save_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        (
            "model_final.pth"
            if is_final
            else "model_best.pth" if is_best else f"checkpoint_{epoch:03d}.pth"
        ),
    )

    logger.info(f"Saving snapshot for epoch {epoch} at {save_path}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": stats,
        },
        save_path,
    )


def train_one_epoch_pixel_classification(
    epoch: int,
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    amp_scaler,
    logger,
    work_dir,
    amp_enabled: bool = True,
    writer: object = None,
    log_interval: int = 1000,
):
    # Set model to training mode
    model.train()
    cur_iter = epoch * len(dataloader)

    end = time.time()

    # Smoothed values for various metrics
    iter_time, data_time = (
        SmoothedValue(fmt="{avg:.4f}"),
        SmoothedValue(fmt="{avg:.4f}"),
    )
    loss_value = SmoothedValue(fmt="{avg:.4f}")

    for batch_id, (targets) in enumerate(dataloader):
        data_time.update(time.time() - end)

        # Move targets to GPU if available
        if torch.cuda.is_available():
            targets = [
                (
                    target.cuda(non_blocking=True)
                    if isinstance(target, torch.Tensor)
                    else target
                )
                for target in targets
            ]

        optimizer.zero_grad()

        with amp.autocast(enabled=amp_enabled):
            # Forward pass
            _, classification_loss = model(targets)

            loss = classification_loss.mean()

        # Backpropagation with AMP
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        # Update metrics
        loss_value.update(loss.item())

        iter_time.update(time.time() - end)
        end = time.time()

        if batch_id % log_interval == 0 or batch_id + 1 == len(dataloader):
            # Log progress
            logger.info(
                f"Epoch [{epoch}][{batch_id}/{len(dataloader)}]\t"
                f"LR {optimizer.param_groups[0]['lr']:.4e}\t"
                f"Loss {loss_value}\t"  # , grad {grad_value})\t"
                f"Time {iter_time} (data {data_time}, eta {datetime.timedelta(seconds=int(iter_time.avg * (len(dataloader) - batch_id - 1)))})\t"
            )

            # Log to Tensorboard if writer is available
            if writer is not None:
                writer.add_scalar(
                    "train/pixel_classification/loss",
                    loss_value.avg,
                    cur_iter + batch_id - 1,
                )

        lr_scheduler.step()

    # Collect training statistics
    train_stats = {"loss": loss_value.avg}

    # Save model snapshot if work_dir is provided
    if work_dir is not None:
        save_snapshot(epoch, model, optimizer, train_stats, work_dir, logger)

    return train_stats


@torch.inference_mode()
def validate_pixel_classification(epoch: int, model, dataloader, logger, writer):
    model.eval()

    loss_value = SmoothedValue(fmt="{avg:.4f}")

    metrics = Metrics(
        len(dataloader.dataset.CLASSIFICATION_CLASSES),
        list(dataloader.dataset.CLASSIFICATION_CLASSES.values()),
        ignore_index=dataloader.dataset.ignore_index,
    )

    with tqdm(
        dataloader,
        desc="Validating",
        leave=False,
        colour="green",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}{postfix}]",
        postfix={
            "loss": 0.0,
        },
    ) as pbar:
        for targets in pbar:
            targets = [
                (
                    target.cuda(non_blocking=True)
                    if isinstance(target, torch.Tensor)
                    else target
                )
                for target in targets
            ]

            preds, classification_loss = model(targets)

            loss = classification_loss.mean()

            loss_value.update(loss.item())

            pred_main = F.interpolate(
                preds[0], scale_factor=8, mode="bilinear", align_corners=False
            )

            metrics.update(pred_main, targets[1])

            pbar.set_postfix(loss=loss_value.avg)

    metrics.collect()

    miou = metrics.metrics["iou"].nanmean().item()

    if writer is not None:
        writer.add_scalar("val/loss", loss_value.avg, epoch)
        writer.add_scalar("val/miou", miou, epoch)

    logger.info(f"=> Validation Metrics:\n\n{metrics}\n")

    return {"loss": loss_value.avg, "miou": miou}


def train_one_epoch_localization(
    epoch: int,
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    amp_scaler,
    logger,
    work_dir,
    amp_enabled: bool = True,
    writer: object = None,
    log_interval: int = 1000,
):
    # Set model to training mode
    model.train()
    cur_iter = epoch * len(dataloader)

    end = time.time()

    # Smoothed values for various metrics
    iter_time, data_time = (
        SmoothedValue(fmt="{avg:.4f}"),
        SmoothedValue(fmt="{avg:.4f}"),
    )
    loss_value = SmoothedValue(fmt="{avg:.4f}")
    centerness_loss_value = SmoothedValue(fmt="{avg:.4f}")
    bbox_loss_value = SmoothedValue(fmt="{avg:.4f}")

    for batch_id, (targets) in enumerate(dataloader):
        data_time.update(time.time() - end)

        # Move targets to GPU if available
        if torch.cuda.is_available():
            targets = [
                (
                    target.cuda(non_blocking=True)
                    if isinstance(target, torch.Tensor)
                    else target
                )
                for target in targets
            ]

        optimizer.zero_grad()

        with amp.autocast(enabled=amp_enabled):
            # Forward pass
            _, (localization_loss, (centerness_loss, bbox_loss)) = model(targets)

            loss = localization_loss.mean()
            centerness_loss = centerness_loss.mean()
            bbox_loss = bbox_loss.mean()

        # Backpropagation with AMP
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        # Update metrics
        loss_value.update(loss.item())
        centerness_loss_value.update(centerness_loss.item())
        bbox_loss_value.update(bbox_loss.item())

        iter_time.update(time.time() - end)
        end = time.time()

        if batch_id % log_interval == 0 or batch_id + 1 == len(dataloader):
            # Log progress
            logger.info(
                f"Epoch [{epoch}][{batch_id}/{len(dataloader)}]\t"
                f"LR {optimizer.param_groups[0]['lr']:.4e}\t"
                f"Loss {loss_value} (centerness_loss {centerness_loss_value}, bbox_loss {bbox_loss_value})\t"  # , grad {grad_value})\t"
                f"Time {iter_time} (data {data_time}, eta {datetime.timedelta(seconds=int(iter_time.avg * (len(dataloader) - batch_id - 1)))})\t"
            )

            # Log to Tensorboard if writer is available
            if writer is not None:
                writer.add_scalar(
                    "train/localization/loss",
                    loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/localization/centerness_loss",
                    centerness_loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/localization/bbox_loss",
                    bbox_loss_value.avg,
                    cur_iter + batch_id - 1,
                )

        lr_scheduler.step()

    # Collect training statistics
    train_stats = {
        "loss": loss_value.avg,
        "centerness_loss": centerness_loss_value.avg,
        "bbox_loss": bbox_loss_value.avg,
    }

    # Save model snapshot if work_dir is provided
    if work_dir is not None:
        save_snapshot(epoch, model, optimizer, train_stats, work_dir, logger)

    return train_stats


@torch.inference_mode()
def validate_localization(epoch: int, model, dataloader, logger, writer):
    model.eval()

    loss_value = SmoothedValue(fmt="{avg:.4f}")
    centerness_loss_value = SmoothedValue(fmt="{avg:.4f}")
    bbox_loss_value = SmoothedValue(fmt="{avg:.4f}")

    mAP = MeanAveragePrecision(iou_type="bbox").cuda()

    with tqdm(
        dataloader,
        desc="Validating",
        leave=False,
        colour="green",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}{postfix}]",
        postfix={
            "loss": 0.0,
        },
    ) as pbar:
        for targets in pbar:
            targets = [
                (
                    target.cuda(non_blocking=True)
                    if isinstance(target, torch.Tensor)
                    else target
                )
                for target in targets
            ]

            preds, (localization_loss, (centerness_loss, bbox_loss)) = model(targets)

            loss = localization_loss.mean()
            centerness_loss = centerness_loss.mean()
            bbox_loss = bbox_loss.mean()

            loss_value.update(loss.item())
            centerness_loss_value.update(centerness_loss.item())
            bbox_loss_value.update(bbox_loss.item())

            detections = model.postprocess(preds, conf_th=0.3)

            pred_det = [
                {
                    "boxes": detections[0][0],
                    "scores": detections[0][1],
                    "labels": detections[0][2],
                }
            ]

            target_det = [{"boxes": targets[2][0], "labels": targets[3][0]}]

            mAP.update(pred_det, target_det)

            pbar.set_postfix(loss=loss_value.avg)

    metrics = mAP.compute()

    if writer is not None:
        writer.add_scalar("val/localization/loss", loss_value.avg, epoch)
        writer.add_scalar(
            "val/localization/centerness_loss", centerness_loss_value.avg, epoch
        )
        writer.add_scalar("val/localization/bbox_loss", bbox_loss_value.avg, epoch)
        writer.add_scalar("val/localization/mAP", metrics["map"].item(), epoch)
        writer.add_scalar("val/localization/mAP50", metrics["map_50"].item(), epoch)
        writer.add_scalar("val/localization/mAP75", metrics["map_75"].item(), epoch)

    logger.info(f"=> Validation Metrics:\n\n{metrics}\n")

    return {
        "loss": loss_value.avg,
        "centerness_loss": centerness_loss_value.avg,
        "bbox_loss": bbox_loss_value.avg,
        "mAP": metrics["map"].item(),
        "mAP50": metrics["map_50"].item(),
        "mAP75": metrics["map_75"].item(),
    }

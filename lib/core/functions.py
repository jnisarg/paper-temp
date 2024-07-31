import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from tqdm import tqdm

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


def train_one_epoch(
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
    (
        loss_value,
        classification_loss_value,
        localization_loss_value,
        centerness_loss_value,
        bbox_loss_value,
        pixel_accuracy_value,
    ) = (SmoothedValue(fmt="{avg:.4f}") for _ in range(6))

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
            (
                _,
                pixel_accuracy,
                classification_loss,
                localization_loss,
                centerness_loss,
                bbox_loss,
            ) = model(targets)

            # Compute mean loss
            # if epoch < 50:
            #     loss = (classification_loss + 0.1 * localization_loss).mean()
            # elif epoch < 100:
            #     loss = (classification_loss + 0.2 * localization_loss).mean()
            # else:
            #     loss = (classification_loss + 0.5 * localization_loss).mean()
            pixel_accuracy = pixel_accuracy.mean()
            loss = (
                classification_loss + pixel_accuracy.item() * localization_loss
            ).mean()
            localization_loss = localization_loss.mean()
            classification_loss = classification_loss.mean()
            centerness_loss = centerness_loss.mean()
            bbox_loss = bbox_loss.mean()

        # Backpropagation with AMP
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        # Update metrics
        loss_value.update(loss.item())
        classification_loss_value.update(classification_loss.item())
        localization_loss_value.update(localization_loss.item())
        pixel_accuracy_value.update(pixel_accuracy.item())
        centerness_loss_value.update(centerness_loss.item())
        bbox_loss_value.update(bbox_loss.item())

        iter_time.update(time.time() - end)
        end = time.time()

        if batch_id % log_interval == 0 or batch_id + 1 == len(dataloader):
            # Log progress
            logger.info(
                f"Epoch [{epoch}][{batch_id}/{len(dataloader)}]\t"
                f"LR {optimizer.param_groups[0]['lr']:.4e}\t"
                f"Loss {loss_value} (classification_loss {classification_loss_value}, localization_loss {localization_loss_value} (centerness_loss {centerness_loss_value}, bbox_loss {bbox_loss_value}), pixel_accuracy {pixel_accuracy_value})\t"  # , grad {grad_value})\t"
                f"Time {iter_time} (data {data_time}, eta {datetime.timedelta(seconds=int(iter_time.avg * (len(dataloader) - batch_id - 1)))})\t"
            )

            # Log to Tensorboard if writer is available
            if writer is not None:
                writer.add_scalar("train/loss", loss_value.avg, cur_iter + batch_id - 1)
                writer.add_scalar(
                    "train/classification_loss",
                    classification_loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/localization_loss",
                    localization_loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/centerness_loss",
                    centerness_loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/bbox_loss",
                    bbox_loss_value.avg,
                    cur_iter + batch_id - 1,
                )
                writer.add_scalar(
                    "train/pixel_accuarcy",
                    pixel_accuracy_value.avg,
                    cur_iter + batch_id - 1,
                )

        lr_scheduler.step()

    # Collect training statistics
    train_stats = {
        "loss": loss_value.avg,
        "classification_loss": classification_loss_value.avg,
        "localization_loss": localization_loss_value.avg,
        "pixel_accuracy": pixel_accuracy_value.avg,
        "centerness_loss": centerness_loss_value.avg,
        "bbox_loss": bbox_loss_value.avg,
    }

    # Save model snapshot if work_dir is provided
    if work_dir is not None:
        save_snapshot(epoch, model, optimizer, train_stats, work_dir, logger)

    return train_stats


@torch.inference_mode()
def validate(epoch: int, model, dataloader, logger, writer):
    model.eval()

    (
        loss_value,
        classification_loss_value,
        localization_loss_value,
        pixel_accuracy_value,
        centerness_loss_value,
        bbox_loss_value,
    ) = (SmoothedValue(fmt="{avg:.4f}") for _ in range(6))

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
            "pixel_accuracy": 0.0,
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

            (
                preds,
                pixel_accuracy,
                classification_loss,
                localization_loss,
                centerness_loss,
                bbox_loss,
            ) = model(targets)

            # if epoch < 50:
            #     loss = (classification_loss + 0.1 * localization_loss).mean()
            # elif epoch < 100:
            #     loss = (classification_loss + 0.2 * localization_loss).mean()
            # else:
            #     loss = (classification_loss + 0.5 * localization_loss).mean()
            loss = (
                classification_loss + pixel_accuracy.item() * localization_loss
            ).mean()
            localization_loss = localization_loss.mean()
            classification_loss = classification_loss.mean()
            pixel_accuracy = pixel_accuracy.mean()
            centerness_loss = centerness_loss.mean()
            bbox_loss = bbox_loss.mean()

            loss_value.update(loss.item())
            classification_loss_value.update(classification_loss.item())
            localization_loss_value.update(localization_loss.item())
            pixel_accuracy_value.update(pixel_accuracy.item())

            metrics.update(preds[0], targets[4])

            pbar.set_postfix(
                loss=loss_value.avg,
                pixel_accuracy=pixel_accuracy_value.avg,
            )

    metrics.collect()

    miou = metrics.metrics["iou"].nanmean().item()

    if writer is not None:
        writer.add_scalar("val/loss", loss_value.avg, epoch)
        writer.add_scalar(
            "train/classification_loss", classification_loss_value.avg, epoch
        )
        writer.add_scalar("train/localization_loss", localization_loss_value.avg, epoch)
        writer.add_scalar("train/centerness_loss", centerness_loss_value.avg, epoch)
        writer.add_scalar("train/bbox_loss", bbox_loss_value.avg, epoch)
        writer.add_scalar("val/pixel_accuracy", pixel_accuracy_value.avg, epoch)
        writer.add_scalar("val/miou", miou, epoch)

        # writer.add_figure(
        #     "val/activations", get_figure(pred[0], pred[1], targets[1]), epoch
        # )

    logger.info(f"=> Validation Metrics:\n\n{metrics}\n")

    return {
        "loss": loss_value.avg,
        "classification_loss": classification_loss_value.avg,
        "localization_loss": localization_loss_value.avg,
        "centerness_loss": centerness_loss_value.avg,
        "bbox_loss": bbox_loss_value.avg,
        "pixel_accuracy": pixel_accuracy_value.avg,
        "miou": miou,
    }

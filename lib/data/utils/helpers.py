from typing import Any, Dict, Optional, Tuple, Union

from torch.utils.data import DataLoader

from lib.data.cityscapes import CityscapesDataset


def build_dataloader(
    config: Dict[str, Any],
    train: bool = True,
    eval_mode: str = "val",
    logger: Optional[Any] = None,
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """
    Build data loaders for training and/or evaluation.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        train (bool): If True, return train and evaluation data loaders.
        eval_mode (str, optional): Evaluation mode ('val' or 'test'). Defaults to 'val'.
        logger (Optional[Any], optional): Logger instance. Defaults to None.

    Returns:
        Union[Tuple[DataLoader, DataLoader], DataLoader]:
            Training and evaluation data loaders if train is True,
            otherwise only the evaluation data loader.
    """
    data_config = config["data"]
    training_config = config["training"]
    num_gpus = len(config["gpus"])

    dataset_kwargs: Dict[str, Any] = {
        "root": data_config["root"],
        "train_size": data_config["train_size"],
        "val_size": data_config["val_size"],
        "mean": data_config["mean"],
        "std": data_config["std"],
        "ignore_index": data_config["ignore_index"],
        "bbox_format": data_config["bbox_format"],
        "logger": logger,
    }

    if train:
        train_dataset = CityscapesDataset(split="train", **dataset_kwargs)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"] * num_gpus,
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        eval_dataset = CityscapesDataset(split=eval_mode, **dataset_kwargs)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_config["batch_size"] * num_gpus,
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, eval_dataloader

    eval_dataset = CityscapesDataset(split=eval_mode, **dataset_kwargs)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config["batch_size"] * num_gpus,
        shuffle=False,
        num_workers=training_config["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    return eval_dataloader

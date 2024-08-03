from typing import Any, Dict, Optional, Tuple, Union

from torch.utils.data import DataLoader

from lib.data.cityscapes import CityscapesPixelClassificationDataset, CityscapesLocalizationDataset


def build_dataloader(
    # config: Dict[str, Any],
    train: bool = True,
    dataset: Union[CityscapesPixelClassificationDataset, CityscapesLocalizationDataset] = None,
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
    # data_config = config["data"]
    # training_config = config["training"]
    # num_gpus = len(config["gpus"])

    dataset_kwargs: Dict[str, Any] = {
        "root": "data/cityscapes",
        "train_size": [1024, 1024],
        "val_size": [1024, 2048],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "ignore_index": 255,
        "bbox_format": "pascal_voc",
        "logger": logger,
    }

    if train:
        train_dataset = dataset(split="train", **dataset_kwargs)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.collate_fn,
        )

        eval_dataset = dataset(split=eval_mode, **dataset_kwargs)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )

        return train_dataloader, eval_dataloader

    eval_dataset = dataset(split=eval_mode, **dataset_kwargs)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
    )

    return eval_dataloader

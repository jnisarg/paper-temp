import os
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataset(Dataset):
    """
    A PyTorch Dataset for the Cityscapes dataset.

    Attributes:
        CLASSIFICATION_CLASSES (Dict[int, str]): Mapping from class indices to class names for classification.
        LOCALIZATION_CLASSES (Dict[int, str]): Mapping from class indices to class names for localization.
        COLOR_PALETTE (Dict[int, List[int]]): Mapping from class indices to RGB color codes.
    """

    CLASSIFICATION_CLASSES: Dict[int, str] = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic_light",
        7: "traffic_sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }

    LOCALIZATION_CLASSES: Dict[int, str] = {
        0: "person",
        1: "rider",
        2: "car",
        3: "truck",
        4: "bus",
        5: "train",
        6: "motorcycle",
        7: "bicycle",
    }

    COLOR_PALETTE: Dict[int, List[int]] = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
    }

    def __init__(
        self,
        root: str,
        split: str,
        train_size: Tuple[int, int],
        val_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        ignore_index: int,
        bbox_format: str = "pascal_voc",
        logger: Optional[Any] = None,
    ) -> None:
        """
        Initialize the CityscapesDataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Dataset split to use ('train', 'val', or 'test').
            train_size (Tuple[int, int]): Size for training images.
            val_size (Tuple[int, int]): Size for validation images.
            mean (Tuple[float, float, float]): Mean for normalization.
            std (Tuple[float, float, float]): Standard deviation for normalization.
            ignore_index (int): Index to ignore in the mask.
            bbox_format (str, optional): Format of bounding boxes. Defaults to 'pascal_voc'.
            logger (Optional[Any], optional): Logger instance. Defaults to None.
        """
        super().__init__()

        self.root = root
        self.train_size = train_size
        self.val_size = val_size
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index
        self.bbox_format = bbox_format
        self.logger = logger
        self.mode = split

        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        with open(os.path.join(root, f"{split}.txt"), "r", encoding="utf-8") as fr:
            self.samples = fr.read().splitlines()

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
                A dictionary containing the image, bounding boxes, labels, mask, and sample information.
        """
        sample = self.samples[index]
        image_path, mask_path, bbox_path = sample.split()

        infos = {"name": os.path.basename(image_path)}

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        bboxes, labels = self._parse_labels(bbox_path)

        infos["raw_height"], infos["raw_width"] = image.shape[:2]

        image, bboxes, labels, mask = self._transforms(image, bboxes, labels, mask)

        image = self._normalize_image(image)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "mask": mask,
            "info": infos,
        }

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from a given path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image.
        """
        image = cv2.imread(os.path.join(self.root, image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """
        Load a mask from a given path.

        Args:
            mask_path (str): Path to the mask file.

        Returns:
            np.ndarray: Loaded mask.
        """
        return cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

    def _parse_labels(self, bbox_path: str) -> Tuple[List[List[int]], List[int]]:
        """
        Parse bounding boxes and labels from a file.

        Args:
            bbox_path (str): Path to the bounding boxes file.

        Returns:
            Tuple[List[List[int]], List[int]]: List of bounding boxes and list of labels.
        """
        with open(os.path.join(self.root, bbox_path), "r", encoding="utf-8") as fr:
            lines = fr.read().splitlines()

        bboxes, labels = [], []
        for line in lines:
            label_id, x1, y1, x2, y2 = map(int, line.split())
            bboxes.append([x1, y1, x2, y2])
            labels.append(label_id)

        return bboxes, labels

    def _transforms(
        self,
        image: np.ndarray,
        bboxes: List[List[int]],
        labels: List[int],
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, List[List[int]], List[int], np.ndarray]:
        """
        Apply transformations to the image, bounding boxes, labels, and mask.

        Args:
            image (np.ndarray): Input image.
            bboxes (List[List[int]]): List of bounding boxes.
            labels (List[int]): List of labels.
            mask (np.ndarray): Input mask.

        Returns:
            Tuple[np.ndarray, List[List[int]], List[int], np.ndarray]:
                Transformed image, bounding boxes, labels, and mask.
        """
        transform = self._get_transform()
        transformed = transform(image=image, bboxes=bboxes, labels=labels, mask=mask)

        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]
        mask = transformed["mask"]

        mask = self._map_mask(mask)

        return image, bboxes, labels, mask

    def _get_transform(self) -> A.Compose:
        """
        Get the transformation pipeline based on the mode (train/val).

        Returns:
            A.Compose: Transformation pipeline.
        """
        if self.mode == "train":
            return A.Compose(
                [
                    A.RandomResizedCrop(
                        height=self.train_size[0], width=self.train_size[1]
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.MotionBlur(p=0.2),
                            A.MedianBlur(blur_limit=3, p=0.1),
                            A.Blur(blur_limit=3, p=0.1),
                        ],
                        p=0.5,
                    ),
                ],
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    min_visibility=0.7,
                    min_area=2000,
                    label_fields=["labels"],
                ),
            )

        return A.Compose(
            [
                A.Resize(height=self.val_size[0], width=self.val_size[1]),
            ],
            bbox_params=A.BboxParams(
                format=self.bbox_format,
                min_visibility=0.7,
                min_area=2000,
                label_fields=["labels"],
            ),
        )

    def _normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Normalize the image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            torch.Tensor: Normalized image.
        """
        image = transforms.ToTensor()(image)
        return transforms.Normalize(mean=self.mean, std=self.std)(image)

    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Map mask values to new values.

        Args:
            mask (np.ndarray): Input mask.

        Returns:
            np.ndarray: Mapped mask.
        """
        temp = mask.copy()
        for k, v in self._get_mask_mapping().items():
            mask[temp == k] = v
        return mask

    def _get_mask_mapping(self) -> Dict[int, int]:
        """
        Get the mapping of mask values to new values.

        Returns:
            Dict[int, int]: Mapping of old mask values to new values.
        """
        return {
            -1: self.ignore_index,
            0: self.ignore_index,
            1: self.ignore_index,
            2: self.ignore_index,
            3: self.ignore_index,
            4: self.ignore_index,
            5: self.ignore_index,
            6: self.ignore_index,
            7: 0,
            8: 1,
            9: self.ignore_index,
            10: self.ignore_index,
            11: 2,
            12: 3,
            13: 4,
            14: self.ignore_index,
            15: self.ignore_index,
            16: self.ignore_index,
            17: 5,
            18: self.ignore_index,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: self.ignore_index,
            30: self.ignore_index,
            31: 16,
            32: 17,
            33: 18,
        }

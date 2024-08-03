import os
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


def gaussian2D(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_umich_gaussian(
    heatmap: np.ndarray, center: Tuple[float, float], radius: int, k: float = 1
) -> np.ndarray:
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO: Debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def gaussian_radius(det_size: Tuple[int, int], min_overlap: float = 0.7) -> float:
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


class CityscapesPixelClassificationDataset(Dataset):

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
        255: [0, 0, 0],
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
    ):
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
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path, mask_path, _ = sample.split()

        info = {"name": os.path.basename(image_path)}

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        image, mask = self._transforms(image, mask)

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, info

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.root, image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        return cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

    def _transforms(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, List[List[int]], List[int], np.ndarray]:
        transform = self._get_transform()
        transformed = transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"]

        mask = self._map_mask(mask)

        return image, mask

    def _get_transform(self) -> A.Compose:
        if self.mode == "train":
            return A.Compose(
                [
                    A.RandomScale(scale_limit=0.5),
                    A.PadIfNeeded(
                        min_height=self.train_size[0],
                        min_width=self.train_size[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=self.ignore_index,
                    ),
                    A.RandomCrop(
                        height=self.train_size[0],
                        width=self.train_size[1],
                        always_apply=True,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )

        return A.Compose(
            [
                A.Resize(
                    height=self.val_size[0],
                    width=self.val_size[1],
                    always_apply=True,
                ),
            ]
        )

    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        temp = mask.copy()
        for k, v in self._get_mask_mapping().items():
            mask[temp == k] = v
        return mask

    def _get_mask_mapping(self) -> Dict[int, int]:
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

    def collate_fn(self, batch):
        images, masks, infos = zip(*batch)

        assert len(images) == len(masks) == len(infos)

        images_list = []
        masks_list = []

        for idx in range(len(images)):
            images_list.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(images[idx])
            )
            masks_list.append(masks[idx])

        images = torch.stack(images_list)
        masks = torch.stack(masks_list)

        return images, masks, infos


class CityscapesLocalizationDataset(CityscapesPixelClassificationDataset):

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

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path, mask_path, bbox_path = sample.split()

        info = {"name": os.path.basename(image_path)}

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        bboxes, labels = self._parse_labels(bbox_path)

        image, bboxes, labels, mask = self._transforms(image, bboxes, labels, mask)

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        bboxes_w, bboxes_h = (
            bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1],
        )

        bbox_centers = np.array(
            [
                (bboxes[..., 0] + bboxes[..., 2]) / 2,
                (bboxes[..., 1] + bboxes[..., 3]) / 2,
            ],
            dtype=np.float32,
        ).T

        bbox_centers[:, 0] = np.clip(bbox_centers[:, 0], 0, image.shape[1] - 1)
        bbox_centers[:, 1] = np.clip(bbox_centers[:, 1], 0, image.shape[0] - 1)

        labels = torch.tensor(labels, dtype=torch.long)

        center_heatmaps = np.zeros(
            (len(self.LOCALIZATION_CLASSES), *image.shape[:2]), dtype=np.float32
        )
        object_mask = torch.ones(len(labels))

        for idx, label in enumerate(labels):
            radius = max(
                0,
                int(gaussian_radius((np.ceil(bboxes_h[idx]), np.ceil(bboxes_w[idx])))),
            )

            bb_ct_int_t = bbox_centers[idx].astype(np.int32)

            if (center_heatmaps[:, bb_ct_int_t[1], bb_ct_int_t[0]] == 1).sum() >= 1.0:
                object_mask[idx] = 0
                continue

            draw_umich_gaussian(center_heatmaps[label], bb_ct_int_t, radius)

            if center_heatmaps[label, bb_ct_int_t[1], bb_ct_int_t[0]] != 1:
                object_mask[idx] = 0

        center_heatmaps = torch.from_numpy(center_heatmaps)

        object_mask = object_mask.eq(1)
        bboxes = bboxes[object_mask]
        labels = labels[object_mask]

        info["bbox_centers"] = torch.tensor(bbox_centers)[object_mask]
        info["object_mask"] = object_mask

        assert (
            center_heatmaps.eq(1).sum().item()
            == len(labels)
            == len(info["bbox_centers"])
        )

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, bboxes, labels, center_heatmaps, info

    def _parse_labels(self, bbox_path: str) -> Tuple[List[List[int]], List[int]]:
        with open(os.path.join(self.root, bbox_path), "r", encoding="utf-8") as fr:
            lines = fr.read().splitlines()

        bboxes, labels = [], []
        for line in lines:
            label_id, x1, x2, y1, y2 = map(int, line.split())
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
        transform = self._get_transform()
        transformed = transform(image=image, bboxes=bboxes, labels=labels, mask=mask)

        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]
        mask = transformed["mask"]

        mask = self._map_mask(mask)

        return image, bboxes, labels, mask

    def _get_transform(self) -> A.Compose:
        if self.mode == "train":
            return A.Compose(
                [
                    A.RandomSizedBBoxSafeCrop(
                        height=self.train_size[0], width=self.train_size[1]
                    ),
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    label_fields=["labels"],
                ),
            )

        return A.Compose(
            [
                A.Resize(height=self.val_size[0], width=self.val_size[1]),
            ],
            bbox_params=A.BboxParams(
                format=self.bbox_format,
                label_fields=["labels"],
            ),
        )

    def collate_fn(self, batch):
        images, masks, bboxes, labels, center_heatmaps, infos = zip(*batch)

        assert len(images) == len(bboxes) == len(labels)

        images_list = []
        masks_list = []
        center_heatmaps_list = []
        bboxes_list = []
        labels_list = []

        max_num = 0
        for idx in range(len(images)):
            images_list.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(images[idx])
            )
            masks_list.append(masks[idx])
            center_heatmaps_list.append(center_heatmaps[idx])

            n = bboxes[idx].shape[0]
            if n > max_num:
                max_num = n

        for idx in range(len(images)):
            bboxes_list.append(
                F.pad(
                    bboxes[idx],
                    (0, 0, 0, max_num - bboxes[idx].shape[0]),
                    value=-1,
                )
            )
            labels_list.append(
                F.pad(
                    labels[idx],
                    (0, max_num - labels[idx].shape[0]),
                    value=-1,
                )
            )

        images = torch.stack(images_list)
        masks = torch.stack(masks_list)
        center_heatmaps = torch.stack(center_heatmaps_list)
        bboxes = torch.stack(bboxes_list)
        labels = torch.stack(labels_list)

        return images, masks, bboxes, labels, center_heatmaps, infos


# class CityscapesDataset(Dataset):

#     CLASSIFICATION_CLASSES: Dict[int, str] = {
#         0: "road",
#         1: "sidewalk",
#         2: "building",
#         3: "wall",
#         4: "fence",
#         5: "pole",
#         6: "traffic_light",
#         7: "traffic_sign",
#         8: "vegetation",
#         9: "terrain",
#         10: "sky",
#         11: "person",
#         12: "rider",
#         13: "car",
#         14: "truck",
#         15: "bus",
#         16: "train",
#         17: "motorcycle",
#         18: "bicycle",
#     }

#     LOCALIZATION_CLASSES: Dict[int, str] = {
#         0: "person",
#         1: "rider",
#         2: "car",
#         3: "truck",
#         4: "bus",
#         5: "train",
#         6: "motorcycle",
#         7: "bicycle",
#     }

#     COLOR_PALETTE: Dict[int, List[int]] = {
#         0: [128, 64, 128],
#         1: [244, 35, 232],
#         2: [70, 70, 70],
#         3: [102, 102, 156],
#         4: [190, 153, 153],
#         5: [153, 153, 153],
#         6: [250, 170, 30],
#         7: [220, 220, 0],
#         8: [107, 142, 35],
#         9: [152, 251, 152],
#         10: [70, 130, 180],
#         11: [220, 20, 60],
#         12: [255, 0, 0],
#         13: [0, 0, 142],
#         14: [0, 0, 70],
#         15: [0, 60, 100],
#         16: [0, 80, 100],
#         17: [0, 0, 230],
#         18: [119, 11, 32],
#     }

#     def __init__(
#         self,
#         root: str,
#         split: str,
#         train_size: Tuple[int, int],
#         val_size: Tuple[int, int],
#         mean: Tuple[float, float, float],
#         std: Tuple[float, float, float],
#         ignore_index: int,
#         bbox_format: str = "pascal_voc",
#         logger: Optional[Any] = None,
#     ) -> None:
#         super().__init__()

#         self.root = root
#         self.train_size = train_size
#         self.val_size = val_size
#         self.mean = mean
#         self.std = std
#         self.ignore_index = ignore_index
#         self.bbox_format = bbox_format
#         self.logger = logger
#         self.mode = split

#         assert split in ["train", "val", "test"], f"Invalid split: {split}"

#         with open(os.path.join(root, f"{split}.txt"), "r", encoding="utf-8") as fr:
#             self.samples = fr.read().splitlines()

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
#         sample = self.samples[index]
#         image_path, mask_path, bbox_path = sample.split()

#         infos = {"name": os.path.basename(image_path)}

#         image = self._load_image(image_path)
#         mask = self._load_mask(mask_path)

#         bboxes, labels = self._parse_labels(bbox_path)

#         infos["raw_height"], infos["raw_width"] = image.shape[:2]

#         image, bboxes, labels, mask = self._transforms(image, bboxes, labels, mask)

#         infos["resize_height"], infos["resize_width"] = image.shape[:2]

#         bboxes = torch.from_numpy(np.array(bboxes)).float()

#         bboxes_w, bboxes_h = (
#             bboxes[..., 2] - bboxes[..., 0],
#             bboxes[..., 3] - bboxes[..., 1],
#         )

#         ct = np.array(
#             [
#                 (bboxes[..., 0] + bboxes[..., 2]) / 2,
#                 (bboxes[..., 1] + bboxes[..., 3]) / 2,
#             ],
#             dtype=np.float32,
#         ).T

#         image = self._normalize_image(image)
#         mask = torch.tensor(mask, dtype=torch.long)
#         labels = torch.tensor(labels, dtype=torch.long)

#         heatmap = np.zeros(
#             (
#                 len(self.LOCALIZATION_CLASSES),
#                 infos["resize_height"],
#                 infos["resize_width"],
#             ),
#             dtype=np.float32,
#         )

#         ct[:, 0] = np.clip(ct[:, 0], 0, infos["resize_width"] - 1)
#         ct[:, 1] = np.clip(ct[:, 1], 0, infos["resize_height"] - 1)

#         infos["gt_heatmap_height"], infos["gt_heatmap_width"] = (
#             infos["resize_height"],
#             infos["resize_width"],
#         )

#         object_mask = torch.ones(len(labels))

#         for idx, label in enumerate(labels):
#             radius = gaussian_radius((np.ceil(bboxes_h[idx]), np.ceil(bboxes_w[idx])))
#             radius = max(0, int(radius))
#             ct_int = ct[idx].astype(np.int32)

#             if (heatmap[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.0:
#                 object_mask[idx] = 0
#                 continue

#             draw_umich_gaussian(heatmap[label], ct_int, radius)

#             if heatmap[label, ct_int[1], ct_int[0]] != 1:
#                 object_mask[idx] = 0

#         heatmap = torch.from_numpy(heatmap)

#         object_mask = object_mask.eq(1)
#         bboxes = bboxes[object_mask]
#         labels = labels[object_mask]

#         infos["ct"] = torch.tensor(ct)[object_mask]
#         infos["object_mask"] = object_mask

#         assert heatmap.eq(1).sum().item() == len(labels) == len(infos["ct"])

#         bboxes = torch.tensor(bboxes, dtype=torch.float32)

#         return image, bboxes, labels, mask, heatmap, infos

#     def collate_fn(self, batch):
#         images, bboxes, labels, masks, heatmaps, infos = zip(*batch)

#         assert len(images) == len(bboxes) == len(labels)

#         images_list = []
#         bboxes_list = []
#         labels_list = []
#         hms_list = []
#         masks_list = []

#         max_num = 0
#         for idx in range(len(images)):
#             images_list.append(images[idx])
#             hms_list.append(heatmaps[idx])
#             masks_list.append(masks[idx])

#             n = bboxes[idx].shape[0]
#             if n > max_num:
#                 max_num = n

#         for idx in range(len(images)):
#             bboxes_list.append(
#                 F.pad(
#                     bboxes[idx],
#                     (0, 0, 0, max_num - bboxes[idx].shape[0]),
#                     value=-1,
#                 )
#             )
#             labels_list.append(
#                 F.pad(
#                     labels[idx],
#                     (0, max_num - labels[idx].shape[0]),
#                     value=-1,
#                 )
#             )

#         images = torch.stack(images_list)
#         bboxes = torch.stack(bboxes_list)
#         labels = torch.stack(labels_list)
#         heatmaps = torch.stack(hms_list)
#         masks = torch.stack(masks_list)

#         return images, bboxes, labels, heatmaps, masks, infos

#     def _load_image(self, image_path: str) -> np.ndarray:
#         image = cv2.imread(os.path.join(self.root, image_path))
#         return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     def _load_mask(self, mask_path: str) -> np.ndarray:
#         return cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

#     def _parse_labels(self, bbox_path: str) -> Tuple[List[List[int]], List[int]]:
#         with open(os.path.join(self.root, bbox_path), "r", encoding="utf-8") as fr:
#             lines = fr.read().splitlines()

#         bboxes, labels = [], []
#         for line in lines:
#             label_id, x1, x2, y1, y2 = map(int, line.split())
#             bboxes.append([x1, y1, x2, y2])
#             labels.append(label_id)

#         return bboxes, labels

#     def _transforms(
#         self,
#         image: np.ndarray,
#         bboxes: List[List[int]],
#         labels: List[int],
#         mask: np.ndarray,
#     ) -> Tuple[np.ndarray, List[List[int]], List[int], np.ndarray]:
#         transform = self._get_transform()
#         transformed = transform(image=image, bboxes=bboxes, labels=labels, mask=mask)

#         image = transformed["image"]
#         bboxes = transformed["bboxes"]
#         labels = transformed["labels"]
#         mask = transformed["mask"]

#         mask = self._map_mask(mask)

#         return image, bboxes, labels, mask

#     def _get_transform(self) -> A.Compose:
#         if self.mode == "train":
#             return A.Compose(
#                 [
#                     A.Resize(height=self.train_size[0], width=self.train_size[1]),
#                     A.HorizontalFlip(p=0.5),
#                 ],
#                 bbox_params=A.BboxParams(
#                     format=self.bbox_format,
#                     label_fields=["labels"],
#                 ),
#             )

#         return A.Compose(
#             [
#                 A.Resize(height=self.val_size[0], width=self.val_size[1]),
#             ],
#             bbox_params=A.BboxParams(
#                 format=self.bbox_format,
#                 label_fields=["labels"],
#             ),
#         )

#     def _normalize_image(self, image: np.ndarray) -> torch.Tensor:
#         image = transforms.ToTensor()(image)
#         return transforms.Normalize(mean=self.mean, std=self.std)(image)

#     def _map_mask(self, mask: np.ndarray) -> np.ndarray:
#         temp = mask.copy()
#         for k, v in self._get_mask_mapping().items():
#             mask[temp == k] = v
#         return mask

#     def _get_mask_mapping(self) -> Dict[int, int]:
#         return {
#             -1: self.ignore_index,
#             0: self.ignore_index,
#             1: self.ignore_index,
#             2: self.ignore_index,
#             3: self.ignore_index,
#             4: self.ignore_index,
#             5: self.ignore_index,
#             6: self.ignore_index,
#             7: 0,
#             8: 1,
#             9: self.ignore_index,
#             10: self.ignore_index,
#             11: 2,
#             12: 3,
#             13: 4,
#             14: self.ignore_index,
#             15: self.ignore_index,
#             16: self.ignore_index,
#             17: 5,
#             18: self.ignore_index,
#             19: 6,
#             20: 7,
#             21: 8,
#             22: 9,
#             23: 10,
#             24: 11,
#             25: 12,
#             26: 13,
#             27: 14,
#             28: 15,
#             29: self.ignore_index,
#             30: self.ignore_index,
#             31: 16,
#             32: 17,
#             33: 18,
#         }

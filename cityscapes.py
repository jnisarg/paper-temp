import os
from typing import Dict, List

import cv2
import numpy as np
import albumentations as A

import lightning as L

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset


class CityscapesDataModule(L.LightningDataModule):

    def __init__(self, batch_size=8):
        super().__init__()

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CityscapesDataset(split="train")
        self.val_dataset = CityscapesDataset(split="eval")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn,
        )


def gaussian2D(shape, sigma: float = 1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_umich_gaussian(heatmap: np.ndarray, center, radius: int, k: float = 1):
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


def gaussian_radius(det_size, min_overlap: float = 0.7) -> float:
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


class CityscapesDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()

        self.root = "./data/cityscapes"
        self.train_size = (1024, 1024)
        self.val_size = (1024, 2048)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.ignore_index = 255
        self.bbox_format = "pascal_voc"

        self.mode = split

        with open(os.path.join(self.root, split + ".txt"), "r") as fr:
            self.samples = fr.read().splitlines()

        # self.transforms = (
        #     A.Compose(
        #         [
        #             # A.RandomScale(scale_limit=0.5),
        #             # A.PadIfNeeded(
        #             #     min_height=1024,
        #             #     min_width=1024,
        #             #     border_mode=cv2.BORDER_CONSTANT,
        #             #     value=0,
        #             #     mask_value=self.ignore_index,
        #             # ),
        #             # A.RandomCrop(height=1024, width=1024),
        #             A.RandomSizedBBoxSafeCrop(height=1024, width=2048),
        #             A.HorizontalFlip(p=0.5),
        #             # A.Normalize(mean=self.mean, std=self.std),
        #         ]
        #     )
        #     if self.mode == "train"
        #     # else A.Compose(
        #     #     [
        #     #         A.PadIfNeeded(
        #     #             min_height=1024,
        #     #             min_width=2048,
        #     #             border_mode=cv2.BORDER_CONSTANT,
        #     #             value=0,
        #     #             mask_value=self.ignore_index,
        #     #         ),
        #     #         A.Normalize(mean=self.mean, std=self.std),
        #     #     ]
        #     # )
        #     else None
        # )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path, mask_path, bbox_path = sample.split()

        info = {"name": os.path.basename(image_path)}

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        bboxes, labels = self._parse_labels(bbox_path)

        image, bboxes, labels, mask = self._transforms(image, bboxes, labels, mask)

        if not bboxes:
            image, mask, bboxes, labels, center_heatmaps, info = self.__getitem__(index)
            return image, mask, bboxes, labels, center_heatmaps, info

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

        center_heatmaps = np.zeros((8, *image.shape[:2]), dtype=np.float32)
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

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.root, image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        return cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

    def _parse_labels(self, bbox_path: str):
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
    ):
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
                    A.Resize(height=384, width=768),
                    A.RandomScale(scale_limit=(0.0, 0.5)),
                    A.RandomCrop(height=384, width=768),
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    label_fields=["labels"],
                ),
            )

        return A.Compose(
            [
                A.Resize(height=384, width=768),
            ],
            bbox_params=A.BboxParams(
                format=self.bbox_format,
                label_fields=["labels"],
            ),
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

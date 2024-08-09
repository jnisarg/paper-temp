import os

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataModule(L.LightningDataModule):

    def __init__(
        self,
        root,
        train_size=(384, 768),
        test_size=(384, 768),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_index=255,
        bbox_down_stride=4,
        bbox_format="pascal_voc",
        transforms_kwargs={},
        eval_transforms_kwargs={},
        train_batch_size=8,
        test_batch_size=1,
        num_workers=8,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.dataset_kwargs = {
            "root": root,
            "train_size": train_size,
            "test_size": test_size,
            "mean": mean,
            "std": std,
            "ignore_index": ignore_index,
            "bbox_down_stride": bbox_down_stride,
            "bbox_format": bbox_format,
            "transforms_kwargs": transforms_kwargs,
            "eval_transforms_kwargs": eval_transforms_kwargs,
        }

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CityscapesDataset(mode="train", **self.dataset_kwargs)

        self.val_dataset = CityscapesDataset(mode="eval", **self.dataset_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn,
        )


class CityscapesDataset(Dataset):

    def __init__(
        self,
        root,
        mode="train",
        train_size=(384, 768),
        test_size=(384, 768),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_index=255,
        bbox_down_stride=4,
        bbox_format="pascal_voc",
        transforms_kwargs={},
        eval_transforms_kwargs={},
    ):
        super().__init__()

        assert mode in ["train", "eval", "test"]

        self.root = root
        self.mode = mode

        self.train_size = train_size
        self.test_size = test_size

        self.mean = mean
        self.std = std

        self.ignore_index = ignore_index

        self.bbox_format = bbox_format
        self.bbox_down_stride = bbox_down_stride

        self.classification_class_names = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            # "rider",
            "car",
            # "truck",
            # "bus",
            # "train",
            # "motorcycle",
            "bicycle",
        ]

        self.localization_class_names = [
            "ped",
            # "rider",
            "car",
            # "truck",
            # "bus",
            # "train",
            # "motorcycle",
            "bicycle",
        ]

        self.color_palette = {
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
            # 12: [255, 0, 0],
            12: [0, 0, 142],
            # 14: [0, 0, 70],
            # 15: [0, 60, 100],
            # 16: [0, 80, 100],
            # 17: [0, 0, 230],
            13: [119, 11, 32],
            255: [0, 0, 0],
        }

        with open(os.path.join(self.root, self.mode + ".txt"), "r") as fr:
            self.samples = fr.read().splitlines()

        if self.mode == "train":
            self.transforms = A.Compose(
                [
                    A.Resize(*self.train_size),
                    A.RandomScale(scale_limit=(0.0, 0.5)),
                    A.RandomCrop(*self.train_size),
                    A.HorizontalFlip(p=0.5),
                ],
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    label_fields=["labels"],
                    **transforms_kwargs,
                ),
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Resize(*self.test_size),
                ],
                bbox_params=A.BboxParams(
                    format=self.bbox_format,
                    label_fields=["labels"],
                    **eval_transforms_kwargs,
                ),
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path, mask_path, bbox_path = sample.split()

        info = {
            "name": os.path.basename(image_path),
            "bbox_down_stride": self.bbox_down_stride,
        }

        image = cv2.imread(os.path.join(self.root, image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

        bboxes, labels = self._parse_bboxes(bbox_path)

        image, mask, bboxes, labels = self._get_transformed_inputs(
            image, mask, bboxes, labels
        )

        if not bboxes:
            idx = np.random.randint(0, len(self))
            return self.__getitem__(idx)

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        bboxes_width, bboxes_height = (
            (bboxes[:, 2] - bboxes[:, 0]) / self.bbox_down_stride,
            (bboxes[:, 3] - bboxes[:, 1]) / self.bbox_down_stride,
        )

        bbox_centers = np.array(
            [
                (bboxes[:, 0] + bboxes[:, 2]) / 2 / self.bbox_down_stride,
                (bboxes[:, 1] + bboxes[:, 3]) / 2 / self.bbox_down_stride,
            ],
            dtype=np.float32,
        ).T

        labels = torch.tensor(labels, dtype=torch.long)

        bbox_centers_heatmaps = np.zeros(
            (
                # len(self.localization_class_names),
                1,
                image.shape[0] // self.bbox_down_stride,
                image.shape[1] // self.bbox_down_stride,
            ),
            dtype=np.float32,
        )

        object_mask = torch.ones(len(labels))

        # print(bboxes_height.shape, bboxes_width.shape)

        for idx, label in enumerate(labels):
            radius = max(
                0,
                int(
                    self.gaussian_radius(
                        (np.ceil(bboxes_height[idx]), np.ceil(bboxes_width[idx]))
                    )
                ),
            )

            bbox_centers_int = bbox_centers[idx].astype(np.int32)

            if (
                bbox_centers_heatmaps[:, bbox_centers_int[1], bbox_centers_int[0]] == 1
            ).sum() >= 1.0:
                object_mask[idx] = 0
                continue

            self.draw_umich_gaussian(bbox_centers_heatmaps[0], bbox_centers_int, radius)

            if bbox_centers_heatmaps[0, bbox_centers_int[1], bbox_centers_int[0]] != 1:
                object_mask[idx] = 0

        bbox_centers_heatmaps = torch.from_numpy(bbox_centers_heatmaps)

        object_mask = object_mask.eq(1)
        bboxes = bboxes[object_mask]
        labels = labels[object_mask]

        info["bbox_centers"] = torch.tensor(bbox_centers, dtype=torch.float)[
            object_mask
        ]
        info["object_mask"] = object_mask

        assert (
            bbox_centers_heatmaps.eq(1).sum().item()
            == len(labels)
            == len(info["bbox_centers"])
        )

        image = transforms.ToTensor()(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, bboxes, labels, bbox_centers_heatmaps, info

    def _parse_bboxes(self, bbox_path):
        with open(os.path.join(self.root, bbox_path), "r") as fr:
            lines = fr.read().splitlines()

        bboxes, labels = [], []

        for line in lines:
            label_id, x1, y1, x2, y2 = map(int, line.split())

            if label_id in [0, 1]:
                label_id = 0
            elif label_id in [2, 3, 4]:
                label_id = 1
            elif label_id in [6, 7]:
                label_id = 2
            else:
                continue

            bboxes.append([x1, y1, x2, y2])
            labels.append(label_id)

        return bboxes, labels

    def _get_transformed_inputs(self, image, mask, bboxes, labels):
        transformed = self.transforms(
            image=image, mask=mask, bboxes=bboxes, labels=labels
        )

        image = transformed["image"]
        mask = transformed["mask"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]

        mask = self._map_mask(mask)

        return image, mask, bboxes, labels

    def _map_mask(self, mask):
        temp = mask.copy()
        for k, v in self._get_mask_mapping().items():
            mask[temp == k] = v
        return mask

    def _get_mask_mapping(self):
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
            25: 11,
            26: 12,
            27: 12,
            28: 12,
            29: self.ignore_index,
            30: self.ignore_index,
            31: self.ignore_index,
            32: 13,
            33: 13,
        }

    def collate_fn(self, batch):
        images, masks, bboxes, labels, bbox_centers_heatmaps, infos = zip(*batch)

        assert len(images) == len(masks) == len(bboxes) == len(labels)

        image_list = []
        mask_list = []
        bbox_list = []
        label_list = []
        bbox_centers_heatmap_list = []

        max_objs = 0

        for idx in range(len(images)):
            image_list.append(
                transforms.Normalize(self.mean, self.std, inplace=True)(images[idx])
            )
            mask_list.append(masks[idx])
            bbox_centers_heatmap_list.append(bbox_centers_heatmaps[idx])

            n = bboxes[idx].shape[0]
            if n > max_objs:
                max_objs = n

        for idx in range(len(images)):
            bbox_list.append(
                F.pad(bboxes[idx], (0, 0, 0, max_objs - bboxes[idx].shape[0]), value=-1)
            )
            label_list.append(
                F.pad(labels[idx], (0, max_objs - labels[idx].shape[0]), value=-1)
            )

        images = torch.stack(image_list)
        masks = torch.stack(mask_list)
        bboxes = torch.stack(bbox_list)
        labels = torch.stack(label_list)
        bbox_centers_heatmaps = torch.stack(bbox_centers_heatmap_list)

        return images, masks, bboxes, labels, bbox_centers_heatmaps, infos

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0

        return h

    @classmethod
    def draw_umich_gaussian(cls, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = cls.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        if (
            min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
        ):  # TODO: Debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    @staticmethod
    def gaussian_radius(det_size, min_overlap=0.7):
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

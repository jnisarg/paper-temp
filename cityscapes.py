import os

import cv2
import albumentations as A

import lightning as L

import torch
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
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )


class CityscapesDataset(Dataset):
    def __init__(self, split="train"):
        super().__init__()

        self.root = "./data/cityscapes"
        self.train_size = (1024, 1024)
        self.val_size = (1024, 2048)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.ignore_index = 255

        self.mode = split

        with open(os.path.join(self.root, split + ".txt"), "r") as fr:
            self.samples = fr.read().splitlines()

        self.transforms = (
            A.Compose(
                [
                    A.RandomScale(scale_limit=0.5),
                    A.PadIfNeeded(
                        min_height=1024,
                        min_width=1024,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=self.ignore_index,
                    ),
                    A.RandomCrop(height=1024, width=1024),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=self.mean, std=self.std),
                ]
            )
            if self.mode == "train"
            else A.Compose(
                [
                    A.PadIfNeeded(
                        min_height=1024,
                        min_width=2048,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=self.ignore_index,
                    ),
                    A.Normalize(mean=self.mean, std=self.std),
                ]
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path, mask_path = sample.split()

        image = cv2.imread(os.path.join(self.root, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

        transformed = self.transforms(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"]

        temp = mask.copy()
        for k, v in self._get_mask_mapping().items():
            mask[temp == k] = v

        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        mask = torch.from_numpy(mask).long()

        return image, mask

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

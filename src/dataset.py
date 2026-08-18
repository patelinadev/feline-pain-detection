"""
dataset.py — CatPainDataset (Step 2)

Loads labeled images from data/pain/ and data/no_pain/ and provides
train/val/test splits with appropriate augmentation.
"""

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CatPainDataset(Dataset):
    """
    Loads images from data/pain/ and data/no_pain/.
    Labels: pain=1, no_pain=0

    Args:
        root_dir: path to data/ folder
        split: 'train', 'val', or 'test'
        transform: torchvision transforms (auto-applied if None)
        seed: random seed for reproducible splits (default=42)
    """

    CLASSES = {"no_pain": 0, "pain": 1}
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    # test = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

    def __init__(self, root_dir, split="train", transform=None, seed=42):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split

        # collect (path, label) pairs
        samples = []
        for class_name, label in self.CLASSES.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    samples.append((os.path.join(class_dir, fname), label))

        # print class counts before split
        pain_total = sum(1 for _, l in samples if l == 1)
        no_pain_total = sum(1 for _, l in samples if l == 0)
        print(f"Dataset loaded: pain={pain_total}, no_pain={no_pain_total}, total={len(samples)}")

        # reproducible shuffle and split
        random.seed(seed)
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * self.TRAIN_RATIO)
        n_val = int(n * self.VAL_RATIO)

        if split == "train":
            self.samples = samples[:n_train]
        elif split == "val":
            self.samples = samples[n_train: n_train + n_val]
        else:
            self.samples = samples[n_train + n_val:]

        print(f"  {split.capitalize()}: {len(self.samples)} samples")

        # transforms: augmentation only on train
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label

"""
KonIQ-10k dataset loader for image quality assessment.

Handles loading, preprocessing, and feature extraction for the KonIQ-10k
benchmark dataset with train/validation/test splits.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from features import extract_all_features


class KonIQDataset(Dataset):
    """
    Dataset for KonIQ-10k image quality assessment benchmark.

    Provides image-parameter-MOS triplets with normalized features
    and configurable augmentation.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 384),
        augment: bool = False,
        param_stats: Optional[Dict[str, np.ndarray]] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            csv_path: Path to KonIQ-10k CSV with scores.
            images_dir: Directory containing images.
            split: One of 'train', 'valid', 'test', or 'all'.
            image_size: Target size (width, height) for resizing.
            augment: Whether to apply data augmentation.
            param_stats: Pre-computed mean/std for feature normalization.
            seed: Random seed for reproducibility.
        """
        assert split in ("train", "valid", "test", "all")

        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.images_dir = Path(images_dir)

        data = pd.read_csv(csv_path)
        self.samples = [
            {"path": self.images_dir / row["image_name"], "mos": row["MOS"]}
            for _, row in data.iterrows()
        ]

        random.Random(seed).shuffle(self.samples)

        if split == "train":
            self.samples = self.samples[:7058]
        elif split == "valid":
            self.samples = self.samples[7058:8058]
        elif split == "test":
            self.samples = self.samples[8058:]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.param_stats = param_stats
        if self.param_stats is None and split in ("train", "all"):
            self.param_stats = self._compute_param_stats()

    def _compute_param_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std of image parameters over the dataset."""
        print("Computing parameter statistics...")
        all_params = []

        for sample in tqdm(self.samples, desc="Extracting features"):
            img = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, self.image_size)
            params = extract_all_features(img)
            all_params.append(params)

        all_params = np.stack(all_params)
        return {
            "mean": np.mean(all_params, axis=0),
            "std": np.std(all_params, axis=0) + 1e-8,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (image, params, mos) tensors.
        """
        sample = self.samples[idx]
        img = cv2.imread(str(sample["path"]), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Cannot read: {sample['path']}")

        img_resized = cv2.resize(img, self.image_size)
        params = extract_all_features(img_resized)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img_rgb)

        if self.param_stats is not None:
            params = (params - self.param_stats["mean"]) / self.param_stats["std"]
        params_tensor = torch.from_numpy(params).float()

        mos = (sample["mos"] - 1) / 4 * 100
        mos_tensor = torch.tensor(mos, dtype=torch.float32)

        return img_tensor, params_tensor, mos_tensor



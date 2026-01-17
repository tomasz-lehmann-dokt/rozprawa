"""
Dataset loaders for moire pattern reduction.

Supports UHDM and TIP2018 datasets with rotating fixed-budget cross-sampling (RFBCS)
for efficient training on imbalanced combined datasets.
"""

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class DemoireDataset(Dataset):
    """
    Combined dataset for UHDM and TIP2018 moire removal benchmarks.
    
    Implements RFBCS strategy: limits TIP2018 samples per epoch while using
    all UHDM samples, with random selection refreshed each epoch.
    """

    def __init__(
        self,
        split: str,
        image_size: int = 256,
        uhdm_dir: str = "",
        tip_dir: str = "",
        tip_budget: int = 15500,
        seed: int = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            split: One of 'train', 'uhdm_test', 'tip_test'.
            image_size: Target size for square crops.
            uhdm_dir: Root directory for UHDM dataset.
            tip_dir: Root directory for TIP2018 dataset.
            tip_budget: Maximum TIP2018 samples per epoch (RFBCS).
            seed: Random seed for reproducibility.
        """
        assert split in ("train", "uhdm_test", "tip_test")

        self.image_size = image_size
        self.moire_paths: List[str] = []
        self.clean_paths: List[str] = []

        if split == "train":
            self._load_train_data(uhdm_dir, tip_dir, tip_budget, seed)
        elif split == "uhdm_test":
            self._load_uhdm_test(uhdm_dir)
        elif split == "tip_test":
            self._load_tip_test(tip_dir)

    def _load_train_data(
        self, uhdm_dir: str, tip_dir: str, tip_budget: int, seed: int
    ) -> None:
        """Load training data from both datasets with RFBCS."""
        uhdm_pairs = []
        tip_pairs = []

        # Load UHDM training data
        for scene_dir in glob.glob(f"{uhdm_dir}/train/*"):
            for moire_path in glob.glob(f"{scene_dir}/*moire.jpg"):
                clean_path = moire_path.replace("moire.jpg", "gt.jpg")
                if Path(clean_path).exists():
                    uhdm_pairs.append((moire_path, clean_path))

        # Load TIP2018 training data
        for source_path in glob.glob(f"{tip_dir}/trainData/*source.png"):
            target_path = source_path.replace("source.png", "target.png")
            if Path(target_path).exists():
                tip_pairs.append((source_path, target_path))

        # Apply RFBCS to TIP2018
        if seed is not None:
            random.seed(seed)
        random.shuffle(tip_pairs)
        tip_pairs = tip_pairs[:tip_budget]

        # Combine and shuffle
        all_pairs = uhdm_pairs + tip_pairs
        random.shuffle(all_pairs)

        self.moire_paths = [p[0] for p in all_pairs]
        self.clean_paths = [p[1] for p in all_pairs]

    def _load_uhdm_test(self, uhdm_dir: str) -> None:
        """Load UHDM test data."""
        for moire_path in glob.glob(f"{uhdm_dir}/test_origin/*moire.jpg"):
            clean_path = moire_path.replace("moire.jpg", "gt.jpg")
            if Path(clean_path).exists():
                self.moire_paths.append(moire_path)
                self.clean_paths.append(clean_path)

    def _load_tip_test(self, tip_dir: str) -> None:
        """Load TIP2018 test data."""
        for source_path in glob.glob(f"{tip_dir}/testData/*source.png"):
            target_path = source_path.replace("source.png", "target.png")
            if Path(target_path).exists():
                self.moire_paths.append(source_path)
                self.clean_paths.append(target_path)

    def __len__(self) -> int:
        return len(self.moire_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (moire_image, clean_image, path) where images are
            normalized to [0, 1] as (C, H, W) tensors.
        """
        moire_img = cv2.imread(self.moire_paths[idx], cv2.IMREAD_COLOR)
        clean_img = cv2.imread(self.clean_paths[idx], cv2.IMREAD_COLOR)

        moire_img = cv2.resize(moire_img, (self.image_size, self.image_size))
        clean_img = cv2.resize(clean_img, (self.image_size, self.image_size))

        moire_tensor = torch.from_numpy(
            np.transpose(moire_img, (2, 0, 1))
        ).float() / 255.0
        clean_tensor = torch.from_numpy(
            np.transpose(clean_img, (2, 0, 1))
        ).float() / 255.0

        return moire_tensor, clean_tensor, self.moire_paths[idx]



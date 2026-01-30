"""
Dataset loaders for low-light image enhancement.

Supports LOL, LOL-v2, and SID datasets with configurable selection strategies.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
import glob


def augment_pair(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentation to image pair."""
    if random.random() > 0.5:
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
    if random.random() > 0.5:
        img1 = cv2.flip(img1, 0)
        img2 = cv2.flip(img2, 0)
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        img1 = ndimage.rotate(img1, angle, reshape=False)
        img2 = ndimage.rotate(img2, angle, reshape=False)
    return img1, img2


def random_crop(
    img1: np.ndarray, img2: np.ndarray, crop_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop both images to target size."""
    h, w, _ = img1.shape
    cw, ch = crop_size
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    return img1[y : y + ch, x : x + cw, :], img2[y : y + ch, x : x + cw, :]


class LowLightDataset(Dataset):
    """
    Combined dataset for low-light enhancement.

    Supports LOL, LOL-v2 (Real_captured, Synthetic), and SID datasets
    with configurable SID exposure selection strategies.
    """

    def __init__(
        self,
        split: str,
        crop_size: Tuple[int, int] = (256, 256),
        datasets_dir: str = "",
        use_lol: bool = True,
        use_lol_v2: bool = True,
        use_sid: bool = True,
        lol_v2_subsets: Optional[List[str]] = None,
        sid_selection: str = "darkest",
        seed: int = 42,
    ) -> None:
        """
        Initialize dataset.

        Args:
            split: One of 'train', 'valid', 'test'.
            crop_size: Target crop size (width, height).
            datasets_dir: Root directory containing all datasets.
            use_lol: Include LOL v1 dataset.
            use_lol_v2: Include LOL v2 dataset.
            use_sid: Include SID dataset.
            lol_v2_subsets: List of ['Real_captured', 'Synthetic'] to include.
            sid_selection: SID exposure selection: 'darkest', 'three_darkest', 'random'.
            seed: Random seed for reproducibility.
        """
        assert split in ("train", "valid", "test")

        self.split = split
        self.crop_size = crop_size
        self.seed = seed
        self.lol_v2_subsets = lol_v2_subsets or ["Real_captured", "Synthetic"]

        self.pairs: List[Tuple[str, str]] = []

        if use_lol:
            self._load_lol(datasets_dir)
        if use_lol_v2:
            self._load_lol_v2(datasets_dir)
        if use_sid:
            self._load_sid(datasets_dir, sid_selection)

    def _load_lol(self, base_dir: str) -> None:
        """Load LOL v1 dataset."""
        lol_dir = Path(base_dir) / "LOL"
        data_dir = lol_dir / ("our485" if self.split == "train" else "eval15")

        low_dir = data_dir / "low"
        high_dir = data_dir / "high"

        for low_path in low_dir.glob("*.png"):
            high_path = high_dir / low_path.name
            if high_path.exists():
                self.pairs.append((str(low_path), str(high_path)))

    def _load_lol_v2(self, base_dir: str) -> None:
        """Load LOL v2 dataset."""
        lol_v2_dir = Path(base_dir) / "LOL-v2"

        for subset in self.lol_v2_subsets:
            subset_dir = lol_v2_dir / subset
            data_dir = subset_dir / ("Train" if self.split == "train" else "Test")

            low_dir = data_dir / "Low"
            high_dir = data_dir / "Normal"

            if not low_dir.exists():
                continue

            if subset == "Synthetic":
                for low_path in low_dir.glob("*.png"):
                    high_path = high_dir / low_path.name
                    if high_path.exists():
                        self.pairs.append((str(low_path), str(high_path)))
            else:
                for low_path in low_dir.glob("low*.png"):
                    id_str = low_path.name.replace("low", "")
                    high_path = high_dir / f"normal{id_str}"
                    if high_path.exists():
                        self.pairs.append((str(low_path), str(high_path)))

    def _load_sid(self, base_dir: str, selection: str) -> None:
        """Load SID dataset with specified exposure selection."""
        sid_dir = Path(base_dir) / "SID"
        part1 = sid_dir / "Dataset_Part1"
        part2 = sid_dir / "Dataset_Part2"

        # 80/20 split
        all_ids = []
        for part_dir in [part1, part2]:
            label_dir = part_dir / "Label"
            if not label_dir.exists():
                continue
            for label_path in label_dir.glob("*.JPG"):
                scene_id = label_path.stem
                all_ids.append((part_dir, scene_id))

        random.seed(self.seed)
        random.shuffle(all_ids)
        split_idx = int(0.8 * len(all_ids))

        if self.split == "train":
            selected_ids = all_ids[:split_idx]
        else:
            selected_ids = all_ids[split_idx:]

        for part_dir, scene_id in selected_ids:
            label_path = part_dir / "Label" / f"{scene_id}.JPG"
            low_dir = part_dir / scene_id

            if not low_dir.exists():
                continue

            low_images = list(low_dir.glob("*.JPG"))
            if not low_images:
                continue

            # Compute brightness for each image
            brightness_list = []
            for img_path in low_images:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    brightness_list.append((str(img_path), img.mean()))

            if not brightness_list:
                continue

            brightness_list.sort(key=lambda x: x[1])

            if selection == "darkest":
                selected_path = brightness_list[0][0]
            elif selection == "three_darkest":
                candidates = brightness_list[:3]
                random.seed(self.seed + int(scene_id))
                selected_path = random.choice(candidates)[0]
            else:  # random
                random.seed(self.seed + int(scene_id))
                selected_path = random.choice(brightness_list)[0]

            self.pairs.append((selected_path, str(label_path)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample.

        Returns:
            Tuple of (low_light, normal, path) tensors normalized to [0, 1].
        """
        low_path, high_path = self.pairs[idx]

        low_img = cv2.imread(low_path, cv2.IMREAD_COLOR)
        high_img = cv2.imread(high_path, cv2.IMREAD_COLOR)

        if low_img is None or high_img is None:
            raise ValueError(f"Failed to load: {low_path}, {high_path}")

        if low_img.shape != high_img.shape:
            # Resize to match
            h, w = min(low_img.shape[0], high_img.shape[0]), min(
                low_img.shape[1], high_img.shape[1]
            )
            low_img = cv2.resize(low_img, (w, h))
            high_img = cv2.resize(high_img, (w, h))

        low_img, high_img = augment_pair(low_img, high_img)
        low_img, high_img = random_crop(low_img, high_img, self.crop_size)

        low_img = low_img.astype(np.float32) / 255.0
        high_img = high_img.astype(np.float32) / 255.0

        low_tensor = torch.from_numpy(np.transpose(low_img, (2, 0, 1)))
        high_tensor = torch.from_numpy(np.transpose(high_img, (2, 0, 1)))

        return low_tensor, high_tensor, low_path

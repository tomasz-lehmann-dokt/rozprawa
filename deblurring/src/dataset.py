"""
HIDE dataset loader for motion deblurring.

HIDE (Human-aware Image DEblurring) contains blurred/sharp image pairs
captured at 240fps. Blur is synthesized by averaging 11 consecutive frames.
Resolution: 1280x720 pixels.
"""

import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HIDEDataset(Dataset):
    """
    Dataset for HIDE motion deblurring benchmark.

    Args:
        root_dir: Path to HIDE dataset root.
        split: One of 'train', 'val', 'test'.
        image_size: Target size (width, height) for resizing.
        limit: Optional limit on number of samples.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        limit: Optional[int] = None,
    ) -> None:
        assert split in ("train", "val", "test")

        self.image_size = image_size
        self.blurred_paths: List[str] = []
        self.sharp_paths: List[str] = []

        gt_dir = os.path.join(root_dir, "GT")

        if split == "train":
            data_dir = os.path.join(root_dir, "train")
            image_list = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        else:
            # Val and test share the test folder, split 80:20
            test_dirs = ["test-close-ups", "test-long-shot"]
            image_list = []
            for subdir in test_dirs:
                pattern = os.path.join(root_dir, "test", subdir, "*.png")
                image_list.extend(sorted(glob.glob(pattern)))

            split_idx = int(0.8 * len(image_list))
            if split == "val":
                image_list = image_list[:split_idx]
            else:
                image_list = image_list[split_idx:]

        if limit is not None:
            image_list = image_list[:limit]

        for img_path in image_list:
            sharp_path = os.path.join(gt_dir, os.path.basename(img_path))
            if os.path.exists(sharp_path):
                self.blurred_paths.append(img_path)
                self.sharp_paths.append(sharp_path)

    def __len__(self) -> int:
        return len(self.sharp_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        blurred = cv2.imread(self.blurred_paths[idx])
        sharp = cv2.imread(self.sharp_paths[idx])

        blurred = cv2.resize(blurred, self.image_size)
        sharp = cv2.resize(sharp, self.image_size)

        # HWC -> CHW, normalize to [0, 1]
        blurred_tensor = (
            torch.from_numpy(np.transpose(blurred, (2, 0, 1))).float() / 255.0
        )
        sharp_tensor = torch.from_numpy(np.transpose(sharp, (2, 0, 1))).float() / 255.0

        return blurred_tensor, sharp_tensor

"""
Image feature extraction for quality assessment.

Computes low-level image statistics used as auxiliary inputs
to the ConvNeXt+MLP quality predictor.
"""

from typing import Tuple

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage import feature


def compute_brightness(image: np.ndarray) -> float:
    """Compute mean brightness from V channel in HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def compute_contrast(image: np.ndarray) -> float:
    """Compute contrast as standard deviation of grayscale intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def compute_sharpness(image: np.ndarray) -> float:
    """Compute sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_colorfulness(image: np.ndarray) -> float:
    """
    Compute colorfulness metric based on opponent color space.

    Reference: Hasler & Süsstrunk (2003).
    """
    b, g, r = cv2.split(image.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def compute_entropy(image: np.ndarray) -> float:
    """Compute Shannon entropy of grayscale histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    return float(scipy_entropy(histogram, base=2))


def compute_edge_density(image: np.ndarray) -> float:
    """Compute proportion of edge pixels using Canny detector."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray)
    return float(edges.mean())


def compute_saturation(image: np.ndarray) -> float:
    """Compute mean saturation from S channel in HSV space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def compute_exposure(image: np.ndarray) -> Tuple[float, float]:
    """
    Compute proportion of under/overexposed pixels.

    Returns:
        Tuple of (underexposed_ratio, overexposed_ratio).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    underexposed = float(np.mean(gray < 50))
    overexposed = float(np.mean(gray > 205))
    return underexposed, overexposed


def extract_all_features(image: np.ndarray) -> np.ndarray:
    """
    Extract all image quality features.

    Args:
        image: BGR image as uint8 numpy array.

    Returns:
        Feature vector of shape (9,) containing:
        [brightness, contrast, sharpness, colorfulness, entropy,
         edge_density, saturation, underexposed, overexposed]
    """
    if image.dtype != np.uint8:
        image = np.uint8(image)

    underexposed, overexposed = compute_exposure(image)

    features = np.array(
        [
            compute_brightness(image),
            compute_contrast(image),
            compute_sharpness(image),
            compute_colorfulness(image),
            compute_entropy(image),
            compute_edge_density(image),
            compute_saturation(image),
            underexposed,
            overexposed,
        ],
        dtype=np.float32,
    )

    return features

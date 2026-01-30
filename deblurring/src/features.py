"""
Image feature extraction for MOS model input.

Computes 9 global image attributes: brightness, contrast, sharpness,
colorfulness, entropy, edge density, saturation, underexposed ratio,
overexposed ratio.
"""

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage import feature


def calculate_brightness(image: np.ndarray) -> float:
    """Mean brightness from V channel of HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def calculate_contrast(image: np.ndarray) -> float:
    """Standard deviation of grayscale intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def calculate_sharpness(image: np.ndarray) -> float:
    """Variance of Laplacian as sharpness measure."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_colorfulness(image: np.ndarray) -> float:
    """Hasler-Süsstrunk colorfulness metric."""
    B, G, R = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_root = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    mean_root = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
    return float(std_root + 0.3 * mean_root)


def calculate_entropy(image: np.ndarray) -> float:
    """Shannon entropy of grayscale histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    return float(scipy_entropy(histogram, base=2))


def calculate_edge_density(image: np.ndarray) -> float:
    """Ratio of edge pixels (Canny detector)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray)
    return float(edges.mean())


def calculate_saturation(image: np.ndarray) -> float:
    """Mean saturation from S channel of HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def calculate_exposure(image: np.ndarray) -> tuple:
    """Ratio of under/overexposed pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    underexposed = float(np.mean(gray < 50))
    overexposed = float(np.mean(gray > 205))
    return underexposed, overexposed


def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract all 9 image quality features.

    Args:
        image: BGR image, uint8 or float [0,1].

    Returns:
        Array of 9 features.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    underexposed, overexposed = calculate_exposure(image)

    return np.array(
        [
            calculate_brightness(image),
            calculate_contrast(image),
            calculate_sharpness(image),
            calculate_colorfulness(image),
            calculate_entropy(image),
            calculate_edge_density(image),
            calculate_saturation(image),
            underexposed,
            overexposed,
        ],
        dtype=np.float32,
    )


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Z-score normalization."""
    return (features - mean) / (std + 1e-8)

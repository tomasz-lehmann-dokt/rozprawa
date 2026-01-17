"""
Evaluation metrics for image quality assessment.

Provides correlation coefficients (PLCC, SROCC) and error metrics (RMSE)
for comparing predicted quality scores against ground truth MOS.
"""

from typing import Tuple
from math import sqrt

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def compute_correlation_metrics(
    predictions: np.ndarray, ground_truth: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute standard IQA evaluation metrics.

    Args:
        predictions: Predicted MOS values.
        ground_truth: Ground truth MOS values.

    Returns:
        Tuple of (SROCC, PLCC, RMSE).
    """
    srocc, _ = stats.spearmanr(predictions, ground_truth)
    plcc, _ = stats.pearsonr(predictions, ground_truth)
    rmse = sqrt(mean_squared_error(ground_truth, predictions))

    return float(srocc), float(plcc), float(rmse)


def compute_range_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    ranges: tuple = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)),
) -> dict:
    """
    Compute RMSE for different MOS quality ranges.

    Useful for analyzing model performance across the full quality spectrum,
    especially in under-represented extreme ranges.

    Args:
        predictions: Predicted MOS values.
        ground_truth: Ground truth MOS values.
        ranges: Tuple of (lower, upper) bounds for each range.

    Returns:
        Dictionary mapping range tuples to RMSE values.
    """
    results = {}
    for lower, upper in ranges:
        mask = (ground_truth >= lower) & (ground_truth < upper)
        if np.any(mask):
            rmse = sqrt(mean_squared_error(ground_truth[mask], predictions[mask]))
        else:
            rmse = float("nan")
        results[(lower, upper)] = rmse

    return results



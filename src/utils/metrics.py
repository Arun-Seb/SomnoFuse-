"""
src/utils/metrics.py
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, cohen_kappa_score
)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def compute_epoch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: Optional[Dict] = None,
) -> Dict:
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":   f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_weighted":f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa":      cohen_kappa_score(y_true, y_pred),
    }
    if extra:
        metrics.update(extra)
    return metrics

"""script to define custom metrics"""
from typing import Optional

import numpy as np
from sklearn.metrics import auc, make_scorer, precision_recall_curve


def custom_recall_score(y_true: np.array, y_score: np.array, threshold: Optional[float] = 0.2) -> float:
    """
    This function implements a custom recall score based on the defined threshold (instead of usual 0.5)

    Args:
        y_true: Real labels
        y_score: Predicted probabilities for the positive target
    Returns:
        The custom recall score

    """
    recall = sum(y_score[y_true == 1] >= threshold) / len(y_score[y_true == 1])
    return recall


def auprc_score(y_true: np.array, y_score: np.array) -> float:
    """
    This function implements the auprc score (area under precision recall curve)

    Args:
        y_true: Real labels
        y_score: Predicted probabilities for the positive target
    Returns:
        The auprc score

    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


custom_recall = make_scorer(custom_recall_score, greater_is_better=True, needs_proba=True)
auprc = make_scorer(auprc_score, greater_is_better=True, needs_proba=True)

CUSTOM_SCORERS = {"auprc": auprc, "custom_recall": custom_recall}

"""
Centralized module for calculating all evaluation metrics.

This keeps the trainer.py clean and makes it easy to add/remove metrics later.
"""

import logging
from typing import Any, Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
)

logger = logging.getLogger(__name__)


def calculate_all_metrics(
    y_true: pd.Series, y_pred: Any, y_prob: Any
) -> Dict[str, float]:
    """
    Calculate all 9 evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : pd.Series
        The actual labels (0 or 1).
    y_pred : Any
        The predicted labels (0 or 1).
    y_prob : Any
        The predicted probabilities for the positive class (stress).

    Returns
    -------
    Dict[str, float]
        A dictionary containing all 9 metrics.
    """
    # First, break down the predictions into True Pos., False Pos., etc.
    # This gives us the basic building blocks for many metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Build a dictionary with all our metrics
    metrics = {
        # Accuracy: What percentage of predictions were correct?
        "accuracy": accuracy_score(y_true, y_pred),
        # Precision: Of all the times we predicted
        # "stress", how often were we right?
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # Recall (Sensitivity): Of all the actual
        # "stress" cases, how many did we catch?
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # F1-Score: A balanced average of Precision and Recall
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        # ROC-AUC: How well does the model separate
        # stressed from non-stressed overall?
        "roc_auc": roc_auc_score(y_true, y_prob),
        # Specificity: Of all the "not stressed" people,
        # how many did we correctly identify?
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        # Sensitivity: Same as Recall (included for clarity)
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        # Balanced Accuracy: Average of Sensitivity and Specificity
        # (good for imbalanced data)
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        # Matthews Correlation Coefficient: A balanced
        # measure even if classes are very imbalanced
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # Log the most important metric (F1-score) for quick feedback
    logger.info(
        f"📊 Calculated 9 metrics. F1-Score: {metrics['f1_score']:.4f}"
    )

    return metrics

"""
Unit tests for the metrics calculation module.

These tests ensure that:
1. All 9 metrics are returned.
2. Metrics are calculated correctly for dummy data.
3. Edge cases (like division by zero) are handled gracefully.
"""

import numpy as np
from src.evaluation.metrics import calculate_all_metrics


class TestCalculateAllMetrics:
    """Test the calculate_all_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics when predictions are 100% correct."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = calculate_all_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_all_wrong_predictions(self):
        """Test metrics when predictions are 100% wrong."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])

        metrics = calculate_all_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["roc_auc"] == 0.0

    def test_single_class_true(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 1])  # 3 TP, 1 FN
        y_prob = np.array([0.9, 0.8, 0.4, 0.7])

        metrics = calculate_all_metrics(y_true, y_pred, y_prob)

        # Precision = TP / (TP + FP). Here FP=0, so Precision is 1.0, not 0.0.
        # Recall = TP / (TP + FN) = 3/4 = 0.75.
        assert metrics["precision"] == 1.0
        assert 0.0 < metrics["recall"] < 1.0

    def test_metric_keys_present(self):
        """Verify all 9 expected metrics are returned."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_prob = np.array([0.3, 0.8, 0.4, 0.4])

        metrics = calculate_all_metrics(y_true, y_pred, y_prob)

        expected_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "specificity",
            "sensitivity",
            "balanced_accuracy",
            "mcc",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

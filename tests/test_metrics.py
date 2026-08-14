"""
Unit tests for the metrics calculation module.

These tests ensure that:
1. All 9 metrics are returned.
2. Metrics are calculated correctly for dummy data.
3. Edge cases (like division by zero) are handled gracefully.
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.metrics import calculate_all_metrics


class TestMetricsCalculation:
    """Test the calculate_all_metrics function."""

    def test_all_metrics_returned(self):
        """Test that the function returns all 9 expected metrics."""
        y_true = pd.Series([1, 0, 1, 0, 1])
        y_pred = [1, 0, 0, 0, 1]
        y_prob = [0.9, 0.1, 0.4, 0.2, 0.8]
        
        metrics = calculate_all_metrics(y_true, y_pred, y_prob)
        
        # List of expected metric names
        expected_metrics = [
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "specificity", "sensitivity", "balanced_accuracy", "mcc"
        ]
        
        # Check that all expected metrics are in the result
        for metric in expected_metrics:
            assert metric in metrics

    def test_perfect_predictions(self):
        """Test metrics when predictions are 100% correct."""
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = [1, 0, 1, 0]
        y_prob = [0.99, 0.01, 0.99, 0.01]
        
        metrics = calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Accuracy and F1 should be 1.0 (perfect)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_zero_division_handling(self):
        """Test that metrics handle cases with no positive predictions gracefully."""
        # True labels have positives, but model predicts ALL zeros
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = [0, 0, 0, 0]
        y_prob = [0.1, 0.1, 0.1, 0.1]
        
        # This should NOT raise an error
        metrics = calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Precision should be 0.0 (not NaN) because we predicted no positives
        assert metrics["precision"] == 0.0
        # Recall should be 0.0 because we missed all positives
        assert metrics["recall"] == 0.0

    def test_metric_values_range(self):
        """Test that all metrics fall within valid ranges [0, 1] (except MCC which is [-1, 1])."""
        y_true = pd.Series([1, 0, 1, 0, 1, 0])
        y_pred = [1, 0, 0, 0, 1, 1]
        y_prob = [0.8, 0.2, 0.4, 0.3, 0.9, 0.6]
        
        metrics = calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Check standard metrics are between 0 and 1
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "specificity", "sensitivity", "balanced_accuracy"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} is out of range"
        
        # MCC is between -1 and 1
        assert -1.0 <= metrics["mcc"] <= 1.0
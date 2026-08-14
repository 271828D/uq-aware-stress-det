"""
Unit tests for the probability extraction utility.

These tests ensure that:
1. Models with predict_proba work correctly.
2. Models without predict_proba (using decision_function) are converted correctly.
3. The output is always a 1D array of probabilities.
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

from src.utils.probabilities import get_prediction_probabilities


class TestProbabilityExtraction:
    """Test the get_prediction_probabilities function."""

    @pytest.fixture
    def dummy_data(self):
        """Create a small dummy dataset for testing."""
        # Generate 100 samples with 5 features each
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        return X, y

    def test_probabilistic_model(self, dummy_data):
        """Test extraction from a model that supports predict_proba (LogisticRegression)."""
        X, y = dummy_data
        
        # Train a Logistic Regression model (supports probabilities)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Get probabilities
        probs = get_prediction_probabilities(model, X)
        
        # Check that we got an array of the correct length
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(y)
        
        # Check that all values are between 0 and 1
        assert np.all((probs >= 0) & (probs <= 1))

    def test_non_probabilistic_model(self, dummy_data):
        """Test extraction from a model that uses decision_function (LinearSVC)."""
        X, y = dummy_data
        
        # Train a LinearSVC model (does NOT support probabilities by default)
        model = LinearSVC(random_state=42)
        model.fit(X, y)
        
        # Verify the model does NOT have predict_proba
        assert not hasattr(model, "predict_proba")
        
        # Get probabilities (should use sigmoid conversion)
        probs = get_prediction_probabilities(model, X)
        
        # Check that we got an array of the correct length
        assert isinstance(probs, np.ndarray)
        assert len(probs) == len(y)
        
        # Check that all values are between 0 and 1 (even though raw scores can be anything)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_output_shape(self, dummy_data):
        """Test that the output is always a 1D array."""
        X, y = dummy_data
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        probs = get_prediction_probabilities(model, X)
        
        # Ensure the result is 1-dimensional (not a matrix)
        assert probs.ndim == 1
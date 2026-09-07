"""
Unit tests for src/utils/probabilities.py.
Verifies sigmoid conversion for models without predict_proba.
"""

import numpy as np
from unittest.mock import MagicMock
from scipy.special import expit
from src.utils.probabilities import get_prediction_probabilities


class TestGetPredictionProbabilities:
    """Test the get_prediction_probabilities utility."""

    def test_model_with_predict_proba(self):
        """Test direct probability extraction for models like RandomForest."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        X = np.array([[1, 2], [3, 4]])

        probs = get_prediction_probabilities(model, X)

        # Should return the second column (index 1)
        expected = np.array([0.7, 0.2])
        np.testing.assert_array_almost_equal(probs, expected)
        model.predict_proba.assert_called_once_with(X)

    def test_model_without_predict_proba(self):
        """Test sigmoid conversion for models like SVC."""
        model = MagicMock()
        # Remove predict_proba to simulate SVC
        del model.predict_proba
        model.decision_function.return_value = np.array([-2.0, 0.0, 2.0])
        X = np.array([[1], [2], [3]])

        probs = get_prediction_probabilities(model, X)

        # Should apply sigmoid (expit) to decision scores
        expected = expit(np.array([-2.0, 0.0, 2.0]))
        np.testing.assert_array_almost_equal(probs, expected)
        model.decision_function.assert_called_once_with(X)

    def test_sigmoid_range(self):
        """Verify that converted probabilities are strictly between 0 and 1."""
        model = MagicMock()
        del model.predict_proba
        # Extreme scores to test sigmoid saturation
        model.decision_function.return_value = np.array([-100, 0, 100])
        X = np.array([[1], [2], [3]])

        probs = get_prediction_probabilities(model, X)

        assert np.all((probs >= 0) & (probs <= 1))
        assert probs[0] < 0.01  # Close to 0
        assert probs[2] > 0.99  # Close to 1

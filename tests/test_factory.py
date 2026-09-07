"""
Unit tests for the ModelFactory class.

These tests ensure that:
1. All models can be instantiated correctly.
2. The is_iterative() method returns correct values.
3. The get_fit_params() method returns appropriate parameters.
"""

import pytest
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.factory import ModelFactory


class TestModelFactoryInstantiation:
    """Test that all models can be created with various parameters."""

    def test_create_svc_with_params(self):
        """Test creating an SVC model with custom parameters."""
        # Create an SVC model with specific settings
        model = ModelFactory.get_model(
            "svc", C=1.5, kernel="rbf", random_state=42
        )

        # Verify it's the correct type
        assert isinstance(model, SVC)
        # Verify the parameters were set correctly
        assert model.C == 1.5
        assert model.kernel == "rbf"
        assert model.random_state == 42

    def test_create_linear_svc(self):
        """Test creating a LinearSVC model."""
        model = ModelFactory.get_model("linear_svc", C=0.5, random_state=123)

        assert isinstance(model, LinearSVC)
        assert model.C == 0.5

    def test_create_logistic_regression(self):
        """Test creating a LogisticRegression model."""
        model = ModelFactory.get_model(
            "logistic_regression", C=2.0, max_iter=500
        )

        assert isinstance(model, LogisticRegression)
        assert model.C == 2.0

    def test_create_random_forest(self):
        """Test creating a RandomForestClassifier model."""
        model = ModelFactory.get_model(
            "random_forest", n_estimators=200, max_depth=10
        )

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 200
        assert model.max_depth == 10

    def test_create_xgboost(self):
        """Test creating an XGBClassifier model."""
        model = ModelFactory.get_model(
            "xgboost", n_estimators=150, learning_rate=0.05
        )

        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 150
        assert model.learning_rate == 0.05

    def test_case_insensitive_model_name(self):
        """Test that model names are case-insensitive."""
        # Should work with uppercase, mixed case, etc.
        model1 = ModelFactory.get_model("SVC")
        model2 = ModelFactory.get_model("SvC")
        model3 = ModelFactory.get_model("svc")

        # All should be SVC instances
        assert isinstance(model1, SVC)
        assert isinstance(model2, SVC)
        assert isinstance(model3, SVC)

    def test_invalid_model_name_raises_error(self):
        """Test that requesting an unknown model raises a ValueError."""
        with pytest.raises(ValueError) as excinfo:
            ModelFactory.get_model("unknown_model")

        # Check that the error message lists available models
        assert "not recognized" in str(excinfo.value)
        assert "svc" in str(excinfo.value)

    def test_null_string_handling(self):
        """Test that 'null' and 'None' strings are converted to None."""
        # When config files send "null" as a string, it should become None
        model = ModelFactory.get_model("svc", C="null", kernel="None")

        assert model.C is None
        assert model.kernel is None


class TestModelFactoryIsIterative:
    """Test the is_iterative() method."""

    def test_xgboost_is_iterative(self):
        """Test that XGBoost is correctly identified as iterative."""
        assert ModelFactory.is_iterative("xgboost") is True
        assert ModelFactory.is_iterative("XGBoost") is True  # Case insensitive

    def test_svc_is_not_iterative(self):
        """Test that SVC is correctly identified as non-iterative."""
        assert ModelFactory.is_iterative("svc") is False
        assert ModelFactory.is_iterative("linear_svc") is False

    def test_random_forest_is_not_iterative(self):
        """Test that RandomForest is correctly identified as non-iterative."""
        assert ModelFactory.is_iterative("random_forest") is False

    def test_logistic_regression_is_not_iterative(self):
        """Test that LogisticRegression is correctly identified as non-iterative."""
        assert ModelFactory.is_iterative("logistic_regression") is False


class TestModelFactoryGetFitParams:
    """Test the get_fit_params() method."""

    def setup_method(self):
        """Create dummy validation data for testing."""
        # Create simple dummy data for validation set
        self.X_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y_val = np.array([0, 1])

    def test_get_fit_params_xgboost_with_early_stopping(self):
        """Test that XGBoost gets eval_set and early_stopping_rounds."""
        params = ModelFactory.get_fit_params(
            model_name="xgboost",
            X_val=self.X_val,
            y_val=self.y_val,
            early_stopping=True,
            patience=5,
            max_epochs=100,
        )

        assert "eval_set" in params
        assert params["early_stopping_rounds"] == 5
        assert params["verbose"] is False

    def test_get_fit_params_xgboost_without_early_stopping(self):
        """Test that XGBoost gets eval_set but NO early_stopping_rounds if disabled."""
        params = ModelFactory.get_fit_params(
            model_name="xgboost",
            X_val=self.X_val,
            y_val=self.y_val,
            early_stopping=False,
            patience=5,
            max_epochs=100,
        )

        assert "eval_set" in params
        assert "early_stopping_rounds" not in params

    def test_get_fit_params_svc_empty_dict(self):
        """Test that SVC (non-iterative) gets an empty dict."""
        params = ModelFactory.get_fit_params(
            model_name="svc",
            X_val=self.X_val,
            y_val=self.y_val,
            early_stopping=True,  # Should be ignored
            patience=5,
            max_epochs=100,
        )

        assert params == {}

    def test_get_fit_params_random_forest_empty_dict(self):
        """Test that RandomForest (non-iterative) gets an empty dict."""
        params = ModelFactory.get_fit_params(
            model_name="random_forest",
            X_val=self.X_val,
            y_val=self.y_val,
            early_stopping=True,
            patience=5,
            max_epochs=100,
        )

        assert params == {}

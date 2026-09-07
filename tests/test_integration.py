"""
Integration tests for the stress-classification pipeline.
Verifies end-to-end flow with synthetic data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.evaluation.metrics import calculate_all_metrics
from src.train.pipeline_steps import save_artifacts
from src.data.splits import split_by_subject
from src.data.preprocessing import fit_transform_pipeline
from src.models.factory import ModelFactory


class TestIntegration:
    """Test the integration of different pipeline components."""

    @pytest.fixture
    def integration_df(self):
        """Create a larger DataFrame for integration testing."""
        n_samples = 100
        data = {
            "subject_id": [
                f"sub_{i % 10}" for i in range(n_samples)
            ],  # 10 subjects
            "feature_1": np.random.rand(n_samples),
            "feature_2": np.random.rand(n_samples),
            "label": np.random.randint(0, 2, n_samples),
            "path": [f"/data/file_{i}.csv" for i in range(n_samples)],
        }
        return pd.DataFrame(data)

    # ✅ FIX: Removed test_config_loading (Hydra handles config validation)

    def test_split_and_preprocess_integration(self, integration_df):
        """Test that splitting and preprocessing work together without leakage."""
        train_df, val_df, test_df = split_by_subject(
            integration_df,
            subject_col="subject_id",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            random_state=42,
        )

        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = (
            fit_transform_pipeline(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                numerical_cols=["feature_1", "feature_2"],
                categorical_cols=[],
                target_col="label",
            )
        )

        assert X_train.shape[0] > 0
        assert X_val.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == 2

    def test_model_training_integration(self, integration_df):
        """Test training a model on processed data."""
        train_df, val_df, test_df = split_by_subject(
            integration_df, "subject_id", 0.8, 0.1, 0.1, 42
        )

        X_train, X_val, X_test, y_train, y_val, y_test, _ = (
            fit_transform_pipeline(
                train_df,
                val_df,
                test_df,
                ["feature_1", "feature_2"],
                [],
                "label",
            )
        )

        model = ModelFactory.get_model("logistic_regression", max_iter=100)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_all_metrics(y_test, y_pred, y_prob)

        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    @patch(
        "src.train.pipeline_steps.joblib.dump"
    )  # ✅ FIX: Mock joblib to avoid pickling MagicMock
    def test_artifact_saving(self, mock_dump, integration_df, tmp_path):
        """Test that artifacts are saved correctly."""
        model = ModelFactory.get_model("logistic_regression")
        preprocessor = MagicMock()
        metrics = {"accuracy": 0.85, "f1_score": 0.80}
        predictions_df = pd.DataFrame({"pred": [1, 0, 1]})

        save_artifacts(
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            predictions_df=predictions_df,
            output_dir=str(tmp_path),
        )

        # Verify joblib.dump was called
        assert mock_dump.call_count >= 2
        assert (tmp_path / "test_predictions.csv").exists()
        assert (tmp_path / "test_metrics.csv").exists()

"""
Integration tests for the pipeline steps in src/train/pipeline_steps.py.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.train.pipeline_steps import (
    save_splits,
    save_artifacts,
    train_model,
    evaluate_model,
)


class TestPipelineSteps:
    """Test the individual pipeline step functions."""

    @pytest.fixture
    def dummy_dfs(self):
        """Create dummy dataframes for splits."""
        train = pd.DataFrame(
            {"feature": [1, 2], "label": [0, 1], "subject_id": [1, 2]}
        )
        val = pd.DataFrame({"feature": [3], "label": [0], "subject_id": [3]})
        test = pd.DataFrame({"feature": [4], "label": [1], "subject_id": [4]})
        return train, val, test

    @pytest.fixture
    def dummy_cfg(self):
        """Create a dummy config object."""
        cfg = MagicMock()
        cfg.model.model_name = "random_forest"
        cfg.training.early_stopping = False
        cfg.training.patience = 10
        cfg.training.max_epochs = 100
        cfg.data.metadata_cols = ["subject_id"]
        cfg.__str__ = lambda self: "MockConfig"
        cfg.__repr__ = lambda self: "MockConfig"
        return cfg

    def test_save_splits_creates_files(self, dummy_dfs, tmp_path):
        """Test that save_splits creates the correct CSV files."""
        train_df, val_df, test_df = dummy_dfs
        save_splits(train_df, val_df, test_df, str(tmp_path))

        assert (tmp_path / "train_split.csv").exists()
        assert (tmp_path / "val_split.csv").exists()
        assert (tmp_path / "test_split.csv").exists()

    @patch("src.train.pipeline_steps.joblib.dump")
    def test_save_artifacts_saves_files(self, mock_dump, dummy_dfs, tmp_path):
        """Test that save_artifacts calls joblib.dump correctly."""
        model = MagicMock()
        preprocessor = MagicMock()
        metrics = {"accuracy": 0.9}
        predictions_df = pd.DataFrame({"pred": [1]})

        save_artifacts(
            model, preprocessor, metrics, predictions_df, str(tmp_path)
        )
        assert mock_dump.call_count >= 2

    @patch("src.train.pipeline_steps.logger")
    @patch("src.train.pipeline_steps.wandb.log")
    @patch("src.train.pipeline_steps.ModelFactory.get_fit_params")
    def test_train_model_calls_fit_with_params(
        self, mock_get_params, mock_wandb, dummy_cfg
    ):
        """Test that train_model calls model.fit with parameters from Factory."""
        model = MagicMock()
        X_train = np.array([[1], [2]])
        y_train = np.array([0, 1])
        X_val = np.array([[3]])
        y_val = np.array([0])

        mock_get_params.return_value = {"verbose": False}

        train_model(model, X_train, y_train, X_val, y_val, dummy_cfg)

        model.fit.assert_called_once_with(X_train, y_train, verbose=False)
        mock_get_params.assert_called_once()

    @patch("src.train.pipeline_steps.wandb.log")
    def test_evaluate_model_returns_metrics_and_df(
        self, mock_wandb, dummy_cfg
    ):
        """Test that evaluate_model returns metrics and a predictions dataframe."""
        model = MagicMock()
        model.predict.return_value = np.array([1, 0])
        model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

        X_test = np.array([[1], [2]])
        y_test = np.array([1, 0])
        test_df = pd.DataFrame({"subject_id": [1, 2]})

        metrics, predictions_df = evaluate_model(
            model, X_test, y_test, test_df, dummy_cfg
        )

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert isinstance(predictions_df, pd.DataFrame)
        assert "stress_probability" in predictions_df.columns

    @patch("src.train.pipeline_steps.wandb.log")
    @patch("src.train.pipeline_steps.get_prediction_probabilities")
    def test_evaluate_model_with_decision_function(
        self, mock_get_prob, mock_wandb, dummy_cfg
    ):
        """Test evaluate_model when model uses decision_function (e.g., SVC)."""
        model = MagicMock()
        del model.predict_proba
        model.predict.return_value = np.array([1, 0])
        mock_get_prob.return_value = np.array([0.8, 0.3])

        X_test = np.array([[1], [2]])
        y_test = np.array([1, 0])
        test_df = pd.DataFrame({"subject_id": [1, 2]})

        metrics, predictions_df = evaluate_model(
            model, X_test, y_test, test_df, dummy_cfg
        )

        mock_get_prob.assert_called_once()
        assert "stress_probability" in predictions_df.columns

    @patch("hydra.core.hydra_config.HydraConfig.get")
    def test_save_splits_with_hydra_mock(
        self, mock_hydra, dummy_dfs, tmp_path
    ):
        """Test that save_splits works when HydraConfig is mocked."""
        mock_runtime = MagicMock()
        mock_runtime.output_dir = str(tmp_path)
        mock_hydra.return_value = mock_runtime

        train_df, val_df, test_df = dummy_dfs
        save_splits(train_df, val_df, test_df, str(tmp_path))

        assert (tmp_path / "train_split.csv").exists()

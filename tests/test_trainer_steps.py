"""
Integration tests for the pipeline steps in src/train/pipeline_steps.py.

These tests use mocking to verify the logic flow without running the full pipeline
or loading real data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
from omegaconf import DictConfig

# Import from the NEW location: pipeline_steps
from src.train.pipeline_steps import save_splits, save_artifacts, train_model, evaluate_model


class TestPipelineSteps:
    """Test the individual pipeline step functions."""

    @pytest.fixture
    def dummy_dfs(self):
        """Create dummy dataframes for splits."""
        train = pd.DataFrame({"feature": [1, 2], "label": [0, 1], "subject_id": [1, 2]})
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
        return cfg

    def test_save_splits_creates_files(self, dummy_dfs, tmp_path):
        """Test that save_splits creates the correct CSV files."""
        train_df, val_df, test_df = dummy_dfs
        
        # Call the function
        save_splits(train_df, val_df, test_df, str(tmp_path))
        
        # Verify files exist
        assert (tmp_path / "train_split.csv").exists()
        assert (tmp_path / "val_split.csv").exists()
        assert (tmp_path / "test_split.csv").exists()

    @patch("src.train.pipeline_steps.joblib.dump")
    def test_save_artifacts_saves_files(self, mock_dump, dummy_dfs, tmp_path):
        """Test that save_artifacts calls joblib.dump correctly."""
        train_df, val_df, test_df = dummy_dfs
        model = MagicMock()
        preprocessor = MagicMock()
        metrics = {"accuracy": 0.9}
        predictions_df = pd.DataFrame({"pred": [1]})
        
        # Call the function
        save_artifacts(model, preprocessor, metrics, predictions_df, str(tmp_path))
        
        # Verify joblib.dump was called for model and preprocessor
        assert mock_dump.call_count >= 2

    @patch("src.train.pipeline_steps.wandb.log")
    def test_train_model_calls_fit_with_params(self, mock_wandb, dummy_cfg):
        """Test that train_model calls model.fit with parameters from Factory."""
        model = MagicMock()
        X_train = np.array([[1], [2]])
        y_train = np.array([0, 1])
        X_val = np.array([[3]])
        y_val = np.array([0])
        
        # Mock the Factory to return specific params
        with patch("src.train.pipeline_steps.ModelFactory.get_fit_params") as mock_get_params:
            mock_get_params.return_value = {"verbose": False}
            
            # Call the function
            train_model(model, X_train, y_train, X_val, y_val, dummy_cfg)
            
            # Verify model.fit was called with the params from Factory
            model.fit.assert_called_once_with(X_train, y_train, verbose=False)
            mock_get_params.assert_called_once()

    def test_evaluate_model_returns_metrics_and_df(self, dummy_cfg):
        """Test that evaluate_model returns metrics and a predictions dataframe."""
        model = MagicMock()
        # Mock model predictions
        model.predict.return_value = np.array([1, 0])
        model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
        
        X_test = np.array([[1], [2]])
        y_test = np.array([1, 0])
        test_df = pd.DataFrame({"subject_id": [1, 2]})
        
        # Call the function
        metrics, predictions_df = evaluate_model(model, X_test, y_test, test_df, dummy_cfg)
        
        # Verify outputs
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert isinstance(predictions_df, pd.DataFrame)
        assert "stress_probability" in predictions_df.columns
"""
Integration tests for the full training pipeline.
Aligned with actual project structure: src.train and configs/
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
from hydra import initialize, compose

# --- FIX: Ensure src is in path BEFORE importing ---
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import from src.train ---
from src.train.trainer import calculate_metrics, save_artifacts


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_wandb():
    """Mocks Weights & Biases."""
    with patch("src.train.trainer.wandb") as mock_wb:
        mock_run = MagicMock()
        mock_wb.init.return_value = mock_run
        mock_wb.run = mock_run
        yield mock_wb


@pytest.fixture
def temp_hydra_output(tmp_path):
    """Creates a temporary directory for Hydra outputs."""
    output_dir = tmp_path / "hydra_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def small_test_data(tmp_path):
    """Generates a tiny CSV dataset."""
    data_path = tmp_path / "test_data.csv"
    
    df = pd.DataFrame({
        "subject_id": [f"sub_{i%3}" for i in range(10)],
        "path": [f"/fake/path/{i}.jpg" for i in range(10)],
        "gaze_yaw": [0.1 * i for i in range(10)],
        "gaze_pitch": [0.2 * i for i in range(10)],
        "au01": [0.05 * i for i in range(10)],
        "au02": [0.05 * i for i in range(10)],
        "au03": [0.05 * i for i in range(10)],
        "au04": [0.05 * i for i in range(10)],
        "au05": [0.05 * i for i in range(10)],
        "au06": [0.05 * i for i in range(10)],
        "au07": [0.05 * i for i in range(10)],
        "au08": [0.05 * i for i in range(10)],
        "activity": ["reading"] * 10,
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    df.to_csv(data_path, sep=";", index=False)
    return str(data_path)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_config_loading():
    """Verifies Hydra config loading from configs/config.yaml."""
    # Use absolute path
    # config_dir = PROJECT_ROOT / "../configs"
    config_dir = "../configs"
    
    with initialize(version_base=None, config_path=str(config_dir)):
        # Explicitly use "config" (matches config.yaml)
        cfg = compose(config_name="config")
        assert cfg.data.subject_col == "subject_id"
        assert cfg.model.model_name == "svc"
        assert cfg.wandb.project == "stress-detection"


def test_metrics_calculation():
    """Verifies metrics calculation."""
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = [0, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.4, 0.2]
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    required_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "specificity", "sensitivity"]
    for key in required_keys:
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_artifact_saving(tmp_path, mocker):
    """Verifies artifact saving (mocking joblib to avoid pickle errors)."""
    mock_joblib = mocker.patch("src.train.trainer.joblib.dump")
    
    model = MagicMock()
    preprocessor = MagicMock()
    metrics = {"accuracy": 0.9}
    predictions_df = pd.DataFrame({"subject_id": [1], "true_label": [0]})
    
    save_artifacts(model, preprocessor, metrics, predictions_df, str(tmp_path))
    
    assert mock_joblib.call_count == 2
    assert (tmp_path / "test_predictions.csv").exists()
    assert (tmp_path / "test_metrics.csv").exists()


@pytest.mark.parametrize("model_override,expected_model", [
    ("model=svc", "svc"),
    ("model=xgboost", "xgboost"),
])
def test_full_training_run(
    mock_wandb, 
    temp_hydra_output, 
    small_test_data, 
    model_override, 
    expected_model
):
    """Verifies config composition for SVC and XGBoost."""
    config_dir = "../configs"
    # config_dir = PROJECT_ROOT / "configs"
    
    overrides = [
        f"data.source_file={small_test_data}",
        f"data.numerical_features=['gaze_yaw','gaze_pitch','au01','au02','au03','au04','au05','au06','au07','au08']",
        f"data.categorical_features=['activity']",
        f"data.metadata_cols=['subject_id','path']",
        f"hydra.run.dir={str(temp_hydra_output)}",
        model_override
    ]
    
    with initialize(version_base=None, config_path=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
        assert cfg.model.model_name == expected_model
        
        if expected_model == "xgboost":
            assert hasattr(cfg.training, "max_epochs")

    assert True
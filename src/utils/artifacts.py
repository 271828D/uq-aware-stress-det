"""
Utility functions for saving and loading ML pipeline artifacts.
"""

import logging
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def save_artifacts(
    output_dir: str | Path,
    model: Any,
    preprocessor: Any,
    metrics: dict[str, float],
    predictions_df: Optional[pd.DataFrame] = None,
    test_split_df: Optional[pd.DataFrame] = None,
    model_name: str = "model",
    save_base_model: bool = False,
    base_model: Optional[Any] = None,
) -> None:
    """
    Save model artifacts, metrics, and predictions to disk.

    Args:
        output_dir: Directory to save artifacts.
        model: The final model (calibrated or base).
        preprocessor: Fitted preprocessor.
        metrics: Dictionary of evaluation metrics.
        predictions_df: DataFrame with predictions (optional).
        test_split_df: DataFrame with test split metadata (optional).
        model_name: Base name for model file.
        save_base_model: If True and base_model provided, also save uncalibrated model.
        base_model: The uncalibrated base model (optional).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving artifacts to {output_path}")

    # 1. Save Main Model
    model_path = output_path / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"   Saved: {model_path.name}")

    # 2. Save Base Model (if calibration was used)
    if save_base_model and base_model is not None:
        base_model_path = output_path / f"{model_name}_base.pkl"
        joblib.dump(base_model, base_model_path)
        logger.info(f"   Saved: {base_model_path.name}")

    # 3. Save Preprocessor
    preproc_path = output_path / "preprocessor.pkl"
    joblib.dump(preprocessor, preproc_path)
    logger.info(f"   Saved: {preproc_path.name}")

    # 4. Save Metrics
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"   Saved: {metrics_path.name}")

    # 5. Save Predictions
    if predictions_df is not None:
        pred_path = output_path / "predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"   Saved: {pred_path.name}")

    # 6. Save Test Split Metadata (for reproducibility)
    if test_split_df is not None:
        test_meta_path = output_path / "test_split_metadata.csv"
        test_split_df.to_csv(test_meta_path, index=False)
        logger.info(f"   Saved: {test_meta_path.name}")

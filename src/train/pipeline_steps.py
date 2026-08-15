"""
Pipeline steps for the stress detection training workflow.

This module contains the individual steps of the training pipeline,
allowing the main trainer.py to remain clean and focused on orchestration.
"""

import logging
import os
from typing import Any, Dict, Tuple

import hydra
import joblib
import pandas as pd
import wandb
from tqdm import tqdm
from omegaconf import DictConfig

from src.data.loader import load_data_from_path
from src.data.splits import split_by_subject
from src.data.preprocessing import fit_transform_pipeline
from src.models.factory import ModelFactory
from src.utils.probabilities import get_prediction_probabilities
from src.evaluation.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save train, validation, and test splits as CSV files for traceability."""
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    train_df.to_csv(train_path, index=False, sep=";")
    val_df.to_csv(val_path, index=False, sep=";")
    test_df.to_csv(test_path, index=False, sep=";")

    logger.info(
        f"✅ Train split saved: {train_path} ({len(train_df)} samples)"
    )
    logger.info(f"✅ Val split saved: {val_path} ({len(val_df)} samples)")
    logger.info(f"✅ Test split saved: {test_path} ({len(test_df)} samples)")


def save_artifacts(
    model: Any,
    preprocessor: Any,
    metrics: Dict[str, float],
    predictions_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save the trained model, preprocessor, and results to the output folder."""
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved to: {model_path}")

    prep_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, prep_path)
    logger.info(f"✅ Preprocessor saved to: {prep_path}")

    pred_path = os.path.join(output_dir, "test_predictions.csv")
    predictions_df.to_csv(pred_path, index=False, sep=";")
    logger.info(f"✅ Predictions saved to: {pred_path}")

    metrics_path = os.path.join(output_dir, "test_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, sep=";")
    logger.info(f"✅ Metrics saved to: {metrics_path}")


def prepare_data(cfg: DictConfig, seed: int) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Any,
    str,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
]:
    """Load, split, and preprocess the data."""
    logger.info("📦 Preparing data...")

    with tqdm(total=1, desc="Loading Data", unit="file") as pbar:
        df = load_data_from_path(cfg.data.source_file)
        pbar.update(1)

    with tqdm(total=1, desc="Splitting Subjects", unit="split") as pbar:
        train_df, val_df, test_df = split_by_subject(
            df,
            subject_col=cfg.data.subject_col,
            train_size=1.0 - cfg.data.test_size - cfg.data.val_size,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            random_state=seed,
        )
        pbar.update(1)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_splits(train_df, val_df, test_df, output_dir)

    with tqdm(total=1, desc="Preprocessing", unit="step") as pbar:
        numerical_features = list(cfg.data.numerical_features)
        categorical_features = list(cfg.data.categorical_features)

        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = (
            fit_transform_pipeline(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                numerical_cols=numerical_features,
                categorical_cols=categorical_features,
                target_col=cfg.data.target_col,
            )
        )
        pbar.update(1)

    return (
        train_df,
        val_df,
        test_df,
        preprocessor,
        output_dir,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def train_model(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    cfg: DictConfig,
) -> Any:
    """Train the model with smart parameter handling."""
    logger.info("🏋️ Training model...")

    fit_params = ModelFactory.get_fit_params(
        model_name=cfg.model.model_name,
        X_val=X_val,
        y_val=y_val,
        early_stopping=cfg.training.early_stopping,
        patience=cfg.training.patience,
        max_epochs=cfg.training.max_epochs,
    )

    with tqdm(total=1, desc="Training Model", unit="epoch") as pbar:
        model.fit(X_train, y_train, **fit_params)
        pbar.update(1)

    val_score = model.score(X_val, y_val)
    logger.info(f"📈 Validation Accuracy: {val_score:.4f}")
    wandb.log({"val_accuracy": val_score})

    return model


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    test_df: pd.DataFrame,
    cfg: DictConfig,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluate the model on the test set."""
    logger.info("📝 Evaluating on test set...")

    with tqdm(total=2, desc="Evaluating", unit="step") as pbar:
        y_pred = model.predict(X_test)
        pbar.update(1)
        y_prob = get_prediction_probabilities(model, X_test)
        pbar.update(1)

    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    wandb.log(metrics)
    logger.info(f"🏆 Final Test Metrics: {metrics}")

    test_metadata = test_df[cfg.data.metadata_cols].reset_index(drop=True)
    predictions_df = pd.concat(
        [
            test_metadata,
            pd.DataFrame(
                {
                    "true_label": y_test,
                    "predicted_label": y_pred,
                    "stress_probability": y_prob,
                }
            ),
        ],
        axis=1,
    )

    return metrics, predictions_df

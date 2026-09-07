"""
Final evaluation pipeline for stress detection.
Retrains the best model and evaluates on the test set with optional calibration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import joblib
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit
from tqdm import tqdm

from src.data.loader import load_data_from_path
from src.data.splits import split_by_subject, split_by_subject_stratified
from src.data.preprocessing import create_preprocessor
from src.models.factory import ModelFactory
from src.evaluation.metrics import calculate_all_metrics
from src.utils.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_best_params(cfg: DictConfig) -> None:
    """
    Load best hyperparameters from optimization_results.yaml and update cfg.

    Strips Hydra CLI prefixes (++, +) from keys and uses force_add=True
    to accept model-specific parameters (e.g., XGBoost n_estimators).

    Args:
        cfg: Hydra DictConfig to update in-place.

    Raises:
        FileNotFoundError: If the optimization_results.yaml path is invalid.
    """
    if not cfg.get("best_params"):
        logger.info(
            "No best_params path provided. Using default config values."
        )
        return

    results_path = Path(cfg.best_params)
    if not results_path.exists():
        raise FileNotFoundError(
            f"optimization_results.yaml not found: {results_path}"
        )

    with open(results_path) as f:
        results = yaml.safe_load(f)

    best_params: Dict[str, Any] = results.get("best_params", {})
    best_value: float = results.get("best_value", float("nan"))

    logger.info(f"📂 Loaded best params from: {results_path}")
    logger.info(f"   Best value: {best_value}")

    for key, value in best_params.items():
        clean_key = key.lstrip("+")
        logger.info(f"   {clean_key}: {value}")
        OmegaConf.update(cfg, clean_key, value, force_add=True)


def get_logits_and_probs(model: Any, X: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract raw logits (decision scores) and probabilities from the model.

    Args:
        model: Trained classifier (base or calibrated).
        X: Processed features (numpy array or sparse matrix).

    Returns:
        Tuple of (logits, probabilities).
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        logger.warning(
            "Model lacks predict_proba. Deriving from decision_function."
        )
        scores = model.decision_function(X)
        probs = expit(scores)

    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        epsilon = 1e-15
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        logger.info("Deriving logits from probabilities (log-odds).")
    else:
        logger.warning(
            "Neither decision_function nor predict_proba available."
        )
        logits = model.predict(X).astype(float)

    return logits, probs


def save_artifacts(
    model: Any,
    preprocessor: Any,
    metrics: Dict[str, float],
    predictions_df: pd.DataFrame,
    test_split_df: pd.DataFrame,
    output_dir: str,
    base_model: Any = None,
) -> None:
    """
    Save the retrained model, preprocessor, metrics, and detailed predictions.

    If base_model is provided, both the calibrated and uncalibrated models
    are saved separately.

    Args:
        model: Final classifier (calibrated or base).
        preprocessor: Fitted ColumnTransformer.
        metrics: Dictionary of evaluation metrics.
        predictions_df: DataFrame with metadata, labels, logits, probabilities.
        test_split_df: Raw test split for traceability.
        output_dir: Hydra output directory path.
        base_model: Uncalibrated base model (optional). If provided, saved separately.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Main model (calibrated if calibration was used)
    model_path = output_path / "final_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"✅ Model saved: {model_path}")

    # Base model (uncalibrated) — only when calibration was applied
    if base_model is not None:
        base_model_path = output_path / "final_model_base.joblib"
        joblib.dump(base_model, base_model_path)
        logger.info(f"✅ Base model saved: {base_model_path}")

    # Preprocessor
    prep_path = output_path / "final_scaler.joblib"
    joblib.dump(preprocessor, prep_path)
    logger.info(f"✅ Preprocessor saved: {prep_path}")

    # Metrics
    metrics_path = output_path / "final_test_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, sep=";")
    logger.info(f"✅ Metrics saved: {metrics_path}")

    # Detailed Predictions
    pred_path = output_path / "final_test_results_detailed.csv"
    predictions_df.to_csv(pred_path, index=False, sep=";")
    logger.info(
        f"✅ Predictions saved: {pred_path} ({len(predictions_df)} rows)"
    )

    # Test Split (traceability)
    split_path = output_path / "final_test_split.csv"
    test_split_df.to_csv(split_path, index=False, sep=";")
    logger.info(f"✅ Test split saved: {split_path}")


class CalibratedModel:
    """
    Wrapper that combines a base classifier with a probability calibrator.

    Compatible with scikit-learn 1.9.0 where cv='prefit' is removed.
    Exposes predict, predict_proba, and decision_function.
    """

    def __init__(self, base_model: Any, calibrator: Any, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def _get_scores(self, X: Any) -> np.ndarray:
        """Get raw scores from the base model."""
        if hasattr(self.base_model, "decision_function"):
            return self.base_model.decision_function(X)
        else:
            return self.base_model.predict_proba(X)[:, 1]

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated class probabilities."""
        scores = self._get_scores(X)
        if self.method == "sigmoid":
            probs = self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            probs = self.calibrator.predict(scores)
        return np.column_stack([1 - probs, probs])

    def decision_function(self, X: Any) -> np.ndarray:
        """Return calibrated log-odds (logit)."""
        probs = self.predict_proba(X)[:, 1]
        epsilon = 1e-15
        probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
        return np.log(probs_clipped / (1 - probs_clipped))


@hydra.main(
    version_base=None, config_path="../../configs", config_name="config"
)
def main(cfg: DictConfig) -> Dict[str, float]:
    """
    Main orchestration: retrain and evaluate on held-out Test.

    Pipeline (calibration mode):
        1. Load best hyperparameters (optional).
        2. Set seed for reproducibility.
        3. Load data and perform subject-aware split.
        4. Train base model on Train only.
        5. Calibrate on Val (subject-aware, no leakage).
        6. Inference on Test set.
        7. Calculate metrics and save all artifacts.

    Pipeline (standard mode):
        1. Load best hyperparameters (optional).
        2. Set seed for reproducibility.
        3. Load data and perform subject-aware split.
        4. Concatenate Train + Val.
        5. Fit preprocessor on Train+Val, transform Test.
        6. Instantiate and retrain model on Train+Val.
        7. Inference on Test set.
        8. Calculate metrics and save all artifacts.

    Args:
        cfg: Hydra DictConfig with all project settings.

    Returns:
        Dictionary of test metrics.
    """
    # 1. Load Best Params (if provided)
    load_best_params(cfg)

    # 2. Set Seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"🌱 Seed set to: {seed}")

    # 3. Load Raw Data
    logger.info("📂 Loading data...")
    df = load_data_from_path(cfg.data.source_file)

    # 4. Subject-Aware Split or Subject-Aware-Stratified-Split
    use_stratified = cfg.data.get("stratified_split", False)

    if use_stratified:
        logger.info("✂️ Performing stratified subject-aware split...")
        train_df, val_df, test_df = split_by_subject_stratified(
            df=df,
            subject_col=cfg.data.subject_col,
            label_col=cfg.data.target_col,
            n_train_subjects=cfg.data.n_train_subjects,
            n_val_subjects=cfg.data.n_val_subjects,
            n_test_subjects=cfg.data.n_test_subjects,
            random_state=seed,
        )
    else:
        logger.info("✂️ Performing subject-aware split...")
        train_df, val_df, test_df = split_by_subject(
            df=df,
            subject_col=cfg.data.subject_col,
            train_size=1.0 - cfg.data.test_size - cfg.data.val_size,
            val_size=cfg.data.val_size,
            test_size=cfg.data.test_size,
            random_state=seed,
        )

    # 5. Determine Calibration Mode
    use_calibration = cfg.data.get("calibrate", False)

    if use_calibration:
        logger.info(
            "⚡ Calibration mode: Train on Train, Calibrate on Val, Test on Test"
        )
        train_df_final = train_df
        val_df_final = val_df
    else:
        logger.info(
            "🔗 Standard mode: Concatenating Train and Val for retraining..."
        )
        train_df_final = pd.concat([train_df, val_df], ignore_index=True)
        val_df_final = None
        logger.info(f"   Train+Val: {len(train_df_final)} samples")

    logger.info(f"   Test: {len(test_df)} samples")

    # 6. Separate Features and Targets
    numerical_cols = list(cfg.data.numerical_features)
    categorical_cols = list(cfg.data.categorical_features)
    target_col = cfg.data.target_col

    X_train_final = train_df_final.drop(columns=[target_col])
    y_train_final = train_df_final[target_col]

    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    X_val_processed = None
    y_val = None
    if use_calibration and val_df_final is not None:
        X_val_raw = val_df_final.drop(columns=[target_col])
        y_val = val_df_final[target_col]
        logger.info(
            f"   Val set for calibration: {len(val_df_final)} samples, "
            f"{val_df_final[cfg.data.subject_col].nunique()} subjects"
        )

    # 7. Preprocess
    logger.info("🧪 Fitting preprocessor...")
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)

    if use_calibration:
        # Fit on Train ONLY — Val must remain untouched for calibration
        X_train_processed = preprocessor.fit_transform(X_train_final)
        X_val_processed = preprocessor.transform(X_val_raw)
        X_test_processed = preprocessor.transform(X_test_raw)
        logger.info(
            "   Preprocessor fit on Train only (Val reserved for calibration)"
        )
    else:
        X_train_processed = preprocessor.fit_transform(X_train_final)
        X_test_processed = preprocessor.transform(X_test_raw)
        logger.info("   Preprocessor fit on Train+Val")

    # 8. Instantiate Model with Best Hyperparameters
    logger.info("🏗️ Instantiating model...")
    model_cfg = cfg.model
    model_name: str = model_cfg.model_name

    model_params: Dict[str, Any] = {
        k: v
        for k, v in model_cfg.items()
        if k not in ("_target_", "model_name")
    }
    model_params.setdefault("random_state", seed)

    base_model = ModelFactory.get_model(model_name, **model_params)

    # 9. Train Base Model
    if use_calibration:
        logger.info("🏋️ Training BASE model on TRAIN set only...")
        # No early stopping — Val is reserved for calibration
        base_model.fit(X_train_processed, y_train_final)
    else:
        logger.info("🏋️ Retraining model on combined Train+Val set...")
        fit_params: Dict[str, Any] = {}
        use_early_stopping = cfg.training.get("early_stopping", False)
        if use_early_stopping and ModelFactory.requires_validation(model_name):
            fit_params["eval_set"] = [(X_test_processed, y_test)]
            fit_params["early_stopping_rounds"] = cfg.training.get(
                "patience", 10
            )
            fit_params["verbose"] = False
            logger.info(
                f"   Early stopping enabled"
                f"(patience={cfg.training.get('patience', 10)})"
            )
        else:
            logger.info(
                "   Training on all Train+Val data (no early stopping)."
            )
        base_model.fit(X_train_processed, y_train_final, **fit_params)

    # 10. Apply Calibration (Manual — sklearn 1.9.0 compatible)
    final_model = base_model
    if use_calibration:
        logger.info(
            "🎯 Calibrating probabilities on Validation set (manual)..."
        )

        # Get raw scores from base model on Val set
        if hasattr(base_model, "decision_function"):
            val_scores = base_model.decision_function(X_val_processed)
        else:
            val_scores = base_model.predict_proba(X_val_processed)[:, 1]

        method = cfg.data.get("calibration_method", "sigmoid")

        if method == "sigmoid":
            calibrator = LogisticRegression(C=1e10)
            calibrator.fit(val_scores.reshape(-1, 1), y_val)
            logger.info("   Fitted Platt scaling (Logistic Regression)")
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_scores, y_val)
            logger.info("   Fitted Isotonic Regression")
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        final_model = CalibratedModel(base_model, calibrator, method)
        logger.info(f"   Manual calibration complete (method='{method}')")

    # 11. Inference on Test Set
    logger.info("🔮 Running inference on Test set...")
    with tqdm(total=3, desc="Inference", unit="step") as pbar:
        y_pred = final_model.predict(X_test_processed)
        pbar.update(1)

        logits, probs = get_logits_and_probs(final_model, X_test_processed)
        pbar.update(1)

        metrics = calculate_all_metrics(
            y_true=y_test, y_pred=y_pred, y_prob=probs
        )
        pbar.update(1)

    # 12. Log Metrics
    logger.info("🏆 Final Test Metrics:")
    for k, v in metrics.items():
        logger.info(f"   {k}: {v:.4f}")

    # 13. Construct Output DataFrame
    metadata_cols = list(cfg.data.get("metadata_cols", []))
    valid_meta_cols = [c for c in metadata_cols if c in test_df.columns]

    predictions_df = pd.DataFrame()
    if valid_meta_cols:
        predictions_df = test_df[valid_meta_cols].reset_index(drop=True)

    predictions_df["true_label"] = y_test.values
    predictions_df["predicted_label"] = y_pred
    predictions_df["logit"] = logits
    predictions_df["pbb_score"] = probs

    # 14. Save Artifacts
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_artifacts(
        model=final_model,
        preprocessor=preprocessor,
        metrics=metrics,
        predictions_df=predictions_df,
        test_split_df=test_df,
        output_dir=output_dir,
        base_model=base_model if use_calibration else None,
    )

    return metrics


if __name__ == "__main__":
    main()

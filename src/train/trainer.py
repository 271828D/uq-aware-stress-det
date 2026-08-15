"""
Main training orchestration script for stress classification.

This script acts as the conductor, coordinating the pipeline steps
defined in pipeline_steps.py to train and evaluate models.
"""

import logging

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.utils.utils import set_seed
from src.train.pipeline_steps import (
    prepare_data,
    train_model,
    evaluate_model,
    save_artifacts,
)
from src.models.factory import ModelFactory

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../configs", config_name="config"
)
def train(cfg: DictConfig) -> float:
    """Main training pipeline orchestrated by Hydra."""

    # 1. Set seed for reproducibility
    set_seed(cfg.seed)

    # 2. Initialize experiment tracking
    wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=cfg.wandb.run_name,
    )

    logger.info("🚀 Starting Training Pipeline...")

    # 3. Prepare Data (Load, Split, Preprocess)
    (
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
    ) = prepare_data(cfg, cfg.seed)

    # 4. Initialize Model
    logger.info("🤖 Initializing model...")
    model = ModelFactory.get_model(
        model_name=cfg.model.model_name,
        **{
            k: v
            for k, v in cfg.model.items()
            if k not in ["_target_", "model_name"]
        },
    )

    # 5. Train Model
    model = train_model(model, X_train, y_train, X_val, y_val, cfg)

    # 6. Evaluate Model
    metrics, predictions_df = evaluate_model(
        model, X_test, y_test, test_df, cfg
    )

    # 7. Save Artifacts
    logger.info("💾 Saving all artifacts...")
    save_artifacts(model, preprocessor, metrics, predictions_df, output_dir)

    logger.info("✅ Training pipeline completed successfully!")

    # 8. Return metric for Optuna
    metric_name = cfg.get("optimization", {}).get("metric", "f1_score")
    optimized_metric = metrics[metric_name]
    logger.info(
        f"🎯 Returning {metric_name} for Optuna: {optimized_metric:.4f}"
    )

    wandb.finish()
    return optimized_metric


if __name__ == "__main__":
    train()

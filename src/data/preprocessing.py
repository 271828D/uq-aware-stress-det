"""
Preprocessing pipeline for stress classification.
Handles scaling, encoding, and imputation without data leakage.
"""

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def create_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    categorical_strategy: str = "onehot",
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer for numerical and categorical features.

    Args:
        numerical_cols: List of numerical feature names.
        categorical_cols: List of categorical feature names.
        categorical_strategy: 'onehot' for SVC/RF, 'ordinal'
        for XGBoost (optional optimization).

    Returns:
        Fitted ColumnTransformer ready to transform data.
    """
    # --- Numerical Pipeline ---
    # 1. Impute missing values with median (robust to outliers)
    # 2. Scale to zero mean and unit variance (critical for SVC, beneficial for others)
    numerical_pipeline = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # --- Categorical Pipeline ---
    if categorical_strategy == "onehot":
        # Best for SVC and Random Forest (handles nominal data, no ordinal assumption)
        categorical_pipeline = Pipeline(
            steps=[
                ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )
    else:
        # Fallback for ordinal encoding if needed later
        from sklearn.preprocessing import OrdinalEncoder

        categorical_pipeline = Pipeline(
            steps=[
                ("cat_imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

    # --- Combine into ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # Drop any columns not explicitly listed (e.g., metadata)
    )

    return preprocessor


def fit_transform_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: str = "label",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ColumnTransformer,
]:
    """
    Fit the preprocessor on TRAIN data only, then transform all splits.

    Args:
        train_df, val_df, test_df: DataFrames from subject-aware split.
        numerical_cols: List of numerical feature column names.
        categorical_cols: List of categorical feature column names.
        target_col: Name of the target column.

    Returns:
        X_train, X_val, X_test (transformed features)
        y_train, y_val, y_test (targets)
        preprocessor (fitted object for inverse transform or saving)
    """
    logger.info(
        f"Creating preprocessor for {len(numerical_cols)} numerical and"
        f"{len(categorical_cols)} categorical features."
    )

    # 1. Create the preprocessor structure
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)

    # 2. Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # 3. FIT on Train, TRANSFORM Train/Val/Test
    # This is the critical step to prevent data leakage
    logger.info("Fitting preprocessor on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)

    logger.info("Transforming validation and test sets...")
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Log shapes to verify consistency
    logger.info(
        f"Processed shapes: Train={X_train_processed.shape},",
        f"Val={X_val_processed.shape}," f"Test={X_test_processed.shape}",
    )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train,
        y_val,
        y_test,
        preprocessor,
    )

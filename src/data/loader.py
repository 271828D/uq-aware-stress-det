"""
Data loading utilities for stress classification.
Handles I/O operations and basic column cleaning.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_data(config: DictConfig) -> pd.DataFrame:
    """
    Load data from path specified in Hydra config.

    Args:
        config: Hydra configuration object containing data.path

    Returns:
        Cleaned pandas DataFrame

    Raises:
        FileNotFoundError: If the specified path does not exist
        ValueError: If required columns are missing
    """
    path = Path(config.data.path)
    return _load_csv(path)


def load_data_from_path(path_str: str) -> pd.DataFrame:
    """
    Load data from a direct string path (useful for CLI or tests).

    Args:
        path_str: String path to the CSV file

    Returns:
        Cleaned pandas DataFrame
    """
    return _load_csv(Path(path_str))


def _load_csv(path: Path) -> pd.DataFrame:
    """Internal helper to handle CSV reading and cleaning."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    logger.info(f"Loading data from {path}")

    # Read CSV with semicolon separator as per your file format
    df = pd.read_csv(path, sep=";")

    # Normalize column names: lowercase, strip whitespace, replace
    # hyphens with underscores. This ensures consistency regardless
    # of how the CSV was saved
    original_cols = df.columns.tolist()
    df.columns = [
        col.strip().lower().replace("-", "_") for col in original_cols
    ]

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    logger.debug(f"Columns: {df.columns.tolist()}")

    return df

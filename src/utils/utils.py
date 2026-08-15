"""
Utility functions for reproducibility and common operations.
Ensures all random sources are seeded for deterministic results.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: Optional[int] = 42) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    This function controls randomness in:
    - Python's built-in random module
    - NumPy's random number generator
    - OS-level hash seed (for consistent dict ordering in Python 3.3+)

    Args:
        seed: Integer seed value. If None, no seeding is applied.

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    if seed is None:
        logger.info("No seed provided, skipping reproducibility setup.")
        return

    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy (used by scikit-learn and most ML libraries)
    np.random.seed(seed)

    # Set environment variable for Python hash seed
    # This ensures dictionary ordering is consistent across runs
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"✅ Random seed set to {seed} for reproducibility.")


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory to create.
    """
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Directory ensured: {directory}")

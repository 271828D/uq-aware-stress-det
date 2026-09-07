"""
Global pytest fixtures for the stress-classifier project.
Provides synthetic data for testing without needing large files.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_subject_df() -> pd.DataFrame:
    """
    Creates a small DataFrame with known subject distribution.

    Structure:
    - 3 Subjects (A, B, C)
    - 2 Samples per subject (6 total rows)
    - Useful for verifying split logic deterministically.
    """
    data = {
        "subject_id": ["A", "A", "B", "B", "C", "C"],
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_synthetic_df() -> pd.DataFrame:
    """
    Creates a larger DataFrame to statistically verify split ratios.
    100 subjects, 10 samples each = 1000 rows.
    """
    n_subjects = 100
    samples_per_subject = 10

    subjects = [
        f"subj_{i}"
        for i in range(n_subjects)
        for _ in range(samples_per_subject)
    ]
    features = list(range(n_subjects * samples_per_subject))

    return pd.DataFrame(
        {
            "subject_id": subjects,
            "feature_val": features,
            "label": [0, 1] * (n_subjects * samples_per_subject // 2),
        }
    )

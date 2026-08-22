"""
Subject-aware splitting and subject-aware stratified splitting
strategies to prevent data leakage. Ensures all samples from a
single subject reside in only one split.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def split_by_subject(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test sets ensuring subject-level separation.

    Uses a two-stage GroupShuffleSplit:
    1. Split all subjects into Train (80%) and Temp (20%).
    2. Split Temp subjects into Val (50% of Temp) and Test (50% of Temp).
       (Resulting in 10% and 10% of total data respectively).

    Args:
        df: Input DataFrame containing subject IDs
        subject_col: Name of the column containing subject identifiers
        train_size: Proportion of subjects for training
        val_size: Proportion of subjects for validation
        test_size: Proportion of subjects for testing
        random_state: Seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If split sizes don't sum to 1.0 or subject_col is missing
    """
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-6:
        raise ValueError(
            f"Split sizes must sum to 1.0. Got {train_size+val_size+test_size}"
        )

    if subject_col not in df.columns:
        raise ValueError(
            f"Subject column '{subject_col}' not found in DataFrame."
            f"Available: {df.columns.tolist()}"
        )

    logger.info(
        f"Starting subject-aware split for {df[subject_col].nunique()} unique subjects"
    )

    # --- Stage 1: Split Train vs (Val + Test) ---
    gss_initial = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )

    # groups=df[subject_col] tells sklearn to keep groups together
    train_idx, temp_idx = next(gss_initial.split(df, groups=df[subject_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # --- Stage 2: Split (Val + Test) into Val and Test ---
    # We need val_size / (val_size + test_size) of the temp set to be validation
    val_ratio_in_temp = val_size / (val_size + test_size)

    gss_temp = GroupShuffleSplit(
        n_splits=1, train_size=val_ratio_in_temp, random_state=random_state
    )

    val_idx, test_idx = next(
        gss_temp.split(temp_df, groups=temp_df[subject_col])
    )

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Log distribution stats
    _log_split_stats(train_df, val_df, test_df, subject_col)

    return train_df, val_df, test_df


def _log_split_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    subject_col: str,
) -> None:
    """Log detailed statistics about the splits for monitoring."""
    logger.info("=== Split Statistics ===")
    logger.info(
        f"Train: {len(train):>5} samples, {train[subject_col].nunique():>3} subjects"
    )
    logger.info(
        f"Val:   {len(val):>5} samples, {val[subject_col].nunique():>3} subjects"
    )
    logger.info(
        f"Test:  {len(test):>5} samples, {test[subject_col].nunique():>3} subjects"
    )

    # Safety check: Ensure no overlap in subjects
    train_subs = set(train[subject_col])
    val_subs = set(val[subject_col])
    test_subs = set(test[subject_col])

    if train_subs & val_subs or train_subs & test_subs or val_subs & test_subs:
        logger.error("CRITICAL: Subject overlap detected between splits!")
    else:
        logger.info(
            "Verification: No subject overlap detected between splits."
        )


def split_by_subject_stratified(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    label_col: str = "label",
    n_train_subjects: int = 42,
    n_val_subjects: int = 5,
    n_test_subjects: int = 6,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test sets with subject-level
    separation and stratification.

    Uses a greedy algorithm to assign entire subjects to splits while attempting to
    preserve the global class distribution (e.g., 57% no-stress / 43% stress) in
    each split.

    Args:
        df: Input DataFrame.
        subject_col: Name of the subject ID column.
        label_col: Name of the label column.
        n_train_subjects: Exact number of subjects for training.
        n_val_subjects: Exact number of subjects for validation.
        n_test_subjects: Exact number of subjects for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        ValueError: If subject columns or label column are missing, or if subject counts
                    do not match the total number of unique subjects.
    """
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    unique_subjects = df[subject_col].unique()
    total_subjects = len(unique_subjects)

    if n_train_subjects + n_val_subjects + n_test_subjects != total_subjects:
        raise ValueError(
            f"Sum subjects split:"
            f"({n_train_subjects}+{n_val_subjects}+{n_test_subjects})"
            f"must equal total unique subjects ({total_subjects})."
        )

    logger.info(
        f"Starting stratified subject-aware split for {total_subjects} unique subjects"
    )

    # 1. Pre-compute stats per subject
    # We need to know the label distribution within each
    # subject to decide where to place it
    subject_stats = []
    for subj in unique_subjects:
        subj_df = df[df[subject_col] == subj]
        n_samples = len(subj_df)
        # Assuming binary classification, calculate ratio of positive class (1)
        n_positive = (subj_df[label_col] == 1).sum()
        positive_ratio = n_positive / n_samples if n_samples > 0 else 0.0

        subject_stats.append(
            {
                "subject": subj,
                "n_samples": n_samples,
                "positive_ratio": positive_ratio,
            }
        )

    # 2. Global positive ratio
    global_positive_ratio = (df[label_col] == 1).mean()
    logger.info(f"Global positive ratio: {global_positive_ratio:.4f}")

    # 3. Greedy Assignment
    # Sort subjects by positive_ratio to help stratification
    # (optional, but helps greedy choice). We shuffle first
    # to ensure random_state affects the initial order, then
    # sort by a "distance" metric or just use the shuffled
    # order and pick the best fit for each slot.

    rng = np.random.RandomState(random_state)
    indices = np.arange(len(subject_stats))
    rng.shuffle(indices)

    # We will assign subjects to 'train', 'val', 'test' one by one.
    # To maximize stratification, at each step, we pick the subject that brings the
    # current split's positive ratio closest to the global ratio.

    splits = {"train": [], "val": [], "test": []}
    remaining_indices = list(indices)

    targets = {
        "train": n_train_subjects,
        "val": n_val_subjects,
        "test": n_test_subjects,
    }

    # Track current sum of positive samples and total samples for
    # each split to calculate running ratio
    current_pos = {"train": 0, "val": 0, "test": 0}
    current_total = {"train": 0, "val": 0, "test": 0}

    # We fill the smaller splits first (val, test) to constrain the larger one (train)
    # This is often better for stratification. Order: val, test, train
    fill_order = ["val", "test", "train"]

    for split_name in fill_order:
        n_needed = targets[split_name]
        for _ in range(n_needed):
            if not remaining_indices:
                break

            best_idx = None
            best_diff = float("inf")

            for idx in remaining_indices:
                subj_data = subject_stats[idx]
                # Hypothetically add this subject to the split
                new_total = current_total[split_name] + subj_data["n_samples"]
                new_pos = current_pos[split_name] + (
                    subj_data["positive_ratio"] * subj_data["n_samples"]
                )
                new_ratio = new_pos / new_total if new_total > 0 else 0

                # How far is this new ratio from the global ratio?
                diff = abs(new_ratio - global_positive_ratio)

                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx

            if best_idx is None:
                raise RuntimeError(
                    "Failed to assign subjects. Check subject counts."
                )

            # Assign the best subject
            subj_data = subject_stats[best_idx]
            splits[split_name].append(subj_data["subject"])
            current_pos[split_name] += (
                subj_data["positive_ratio"] * subj_data["n_samples"]
            )
            current_total[split_name] += subj_data["n_samples"]

            # Remove from remaining
            remaining_indices.remove(best_idx)

            # Log progress
            logger.debug(
                f"Assigned subject {subj_data['subject']} to {split_name}. "
                f"Split ratio: {current_pos[split_name]/current_total[split_name]:.4f}"
            )

    # 4. Create DataFrames
    train_df = df[df[subject_col].isin(splits["train"])].reset_index(drop=True)
    val_df = df[df[subject_col].isin(splits["val"])].reset_index(drop=True)
    test_df = df[df[subject_col].isin(splits["test"])].reset_index(drop=True)

    # 5. Verify and Log
    _log_split_stats_stratified(
        train_df, val_df, test_df, subject_col, label_col
    )

    return train_df, val_df, test_df


def _log_split_stats_stratified(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    subject_col: str,
    label_col: str,
) -> None:
    """Log statistics including label distribution for stratified splits."""
    logger.info("=== Stratified Split Statistics ===")
    for name, df in zip(["Train", "Val", "Test"], [train, val, test]):
        n_subs = df[subject_col].nunique()
        n_samples = len(df)
        pos_ratio = (df[label_col] == 1).mean() if n_samples > 0 else 0
        logger.info(
            f"{name:>5}: {n_samples:>6} samples,"
            f"{n_subs:>3} subjects, Pos Ratio: {pos_ratio:.4f}"
        )

    # Safety check: Ensure no overlap
    train_subs = set(train[subject_col])
    val_subs = set(val[subject_col])
    test_subs = set(test[subject_col])

    if train_subs & val_subs or train_subs & test_subs or val_subs & test_subs:
        logger.error("CRITICAL: Subject overlap detected in stratified split!")
    else:
        logger.info("Verification: No subject overlap detected.")

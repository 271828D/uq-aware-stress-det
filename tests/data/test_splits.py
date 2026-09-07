"""
Tests for src/data/splits.py
"""

import pytest
from src.data.splits import split_by_subject


class TestSubjectSplit:
    def test_split_no_overlap(self, sample_subject_df):
        """Ensure no subject ID appears in more than one split."""
        train, val, test = split_by_subject(
            sample_subject_df,
            subject_col="subject_id",
            train_size=0.6,  # Adjusted for small sample (3 subjects -> ~2 train, 1 val, 1 test logic is hard with 3, so we check overlap primarily) # noqa
            val_size=0.2,
            test_size=0.2,
            random_state=42,
        )

        train_subs = set(train["subject_id"])
        val_subs = set(val["subject_id"])
        test_subs = set(test["subject_id"])

        # Assert no intersection
        assert train_subs.isdisjoint(val_subs), "Train and Val share subjects!"
        assert train_subs.isdisjoint(
            test_subs
        ), "Train and Test share subjects!"
        assert val_subs.isdisjoint(test_subs), "Val and Test share subjects!"

    def test_split_ratios_large(self, large_synthetic_df):
        """Verify split sizes are within 5% of target on a larger dataset."""
        train, val, test = split_by_subject(
            large_synthetic_df,
            subject_col="subject_id",
            train_size=0.8,
            val_size=0.1,
            test_size=0.1,
            random_state=42,
        )

        total = len(large_synthetic_df)

        # Allow 5% tolerance due to integer rounding of subjects
        assert abs(len(train) / total - 0.8) < 0.05
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05

    def test_invalid_split_sum(self, sample_subject_df):
        """Ensure error is raised if sizes don't sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_by_subject(
                sample_subject_df, train_size=0.5, val_size=0.5, test_size=0.5
            )

    def test_missing_subject_col(self, sample_subject_df):
        """Ensure error is raised if subject column is missing."""
        with pytest.raises(ValueError, match="not found in DataFrame"):
            split_by_subject(sample_subject_df, subject_col="non_existent_col")

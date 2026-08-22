"""
Tests for src/data/loader.py
"""

import pytest
import pandas as pd

# Import the internal helper to test it directly or mock the Path
from src.data.loader import _load_csv, load_data_from_path


class TestDataLoader:
    def test_column_cleaning(self, tmp_path):
        """Test that column names are properly lowercased and cleaned."""
        # Create a real temporary file for this integration-style test
        csv_content = "Subject-ID;Gaze_Yaw;Label\nA;1.0;0\nB;2.0;1"
        file_path = tmp_path / "test.csv"
        file_path.write_text(csv_content)

        df = load_data_from_path(str(file_path))

        assert "subject_id" in df.columns
        assert "gaze_yaw" in df.columns
        assert "Subject-ID" not in df.columns

    def test_file_not_found(self):
        """Test FileNotFoundError is raised for missing paths."""
        with pytest.raises(FileNotFoundError):
            load_data_from_path("/non/existent/path.csv")

    def test_load_with_mock(self, mocker):
        """
        Mock pandas.read_csv to verify we pass the correct separator.
        This ensures we don't accidentally break the ';' requirement.
        """
        mock_df = pd.DataFrame({"col": [1]})
        mock_read = mocker.patch("pandas.read_csv", return_value=mock_df)

        # Patch Path.exists to return True so we pass the file check
        mocker.patch("pathlib.Path.exists", return_value=True)

        _load_csv(mocker.MagicMock())

        # Assert read_csv was called with semicolon separator
        mock_read.assert_called_once()
        assert mock_read.call_args.kwargs["sep"] == ";"

    def test_load_data_from_config(self, mocker, tmp_path):
        """Test loading data using a Hydra DictConfig object."""
        # Create a real temp file
        csv_content = "subject_id;label\nA;1"
        file_path = tmp_path / "config_test.csv"
        file_path.write_text(csv_content)

        # Mock a Hydra DictConfig
        mock_config = mocker.MagicMock()
        mock_config.data.path = str(file_path)

        # Import the function we haven't tested yet
        from src.data.loader import load_data

        df = load_data(mock_config)

        assert len(df) == 1
        assert "subject_id" in df.columns

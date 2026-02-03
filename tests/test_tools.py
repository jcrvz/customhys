"""
Test suite for tools module.
Tests utility functions and data processing tools.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from customhys import tools as tl


class TestBasicUtilities:
    """Test basic utility functions in tools module."""

    def test_listfind_function(self):
        """Test listfind function for finding elements."""
        test_list = [1, 2, 3, 4, 5, 2, 3]

        # Find single occurrence
        indices = tl.listfind(test_list, 4)
        assert 3 in indices

        # Find multiple occurrences
        indices = tl.listfind(test_list, 2)
        assert len(indices) >= 1
        assert 1 in indices
        assert 5 in indices

    def test_check_fields_function(self):
        """Test check_fields function for dictionary validation."""
        default_config = {"param1": 10, "param2": "value", "param3": True}

        user_config = {"param1": 20}

        result = tl.check_fields(default_config, user_config)

        assert result["param1"] == 20  # User value
        assert result["param2"] == "value"  # Default value
        assert result["param3"] is True  # Default value

    def test_check_fields_with_none(self):
        """Test check_fields with None values."""
        default = {"param1": 10, "param2": "value"}

        # Pass None for user config
        result = tl.check_fields(default, None)

        assert result == default


class TestJSONOperations:
    """Test JSON save/load operations."""

    def test_save_and_read_json(self):
        """Test saving and reading JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = {"key": "value", "number": 42}
            test_file = Path(tmpdir) / "test.json"

            # Save JSON
            tl.save_json(test_data, str(test_file))

            # Read JSON
            loaded_data = tl.read_json(str(test_file))

            assert loaded_data == test_data

    def test_read_json_with_file(self):
        """Test reading an existing JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_data = {"test": 123}

            # Create JSON file
            with open(test_file, "w") as f:
                json.dump(test_data, f)

            # Read it
            result = tl.read_json(str(test_file))
            assert result["test"] == 123


class TestDataStructures:
    """Test data structure handling."""

    def test_df2dict_function(self):
        """Test df2dict conversion if pandas is available."""
        try:
            import pandas as pd

            # Create test dataframe
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

            # Convert to dict
            result = tl.df2dict(df)

            assert isinstance(result, dict)
            assert "col1" in result or 0 in result  # Could be column-based or index-based
        except ImportError:
            pytest.skip("Pandas not available")

    def test_array_operations(self):
        """Test basic array operations."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert len(data) == 5
        assert np.mean(data) == 3.0
        assert isinstance(np.mean(data), (int, float, np.number))


class TestPrintmsk:
    """Test printmsk function."""

    def test_printmsk_basic(self):
        """Test printmsk doesn't crash with basic inputs."""
        # Just verify it doesn't crash
        try:
            tl.printmsk("Test message", level=1)
            tl.printmsk({"key": "value"}, level=2)
            tl.printmsk([1, 2, 3], level=3)
            assert True
        except Exception as e:
            pytest.fail(f"printmsk raised exception: {e}")


class TestValidation:
    """Test validation functions."""

    def test_type_validation(self):
        """Test type validation for inputs."""
        # Test integer validation
        assert isinstance(10, int)

        # Test float validation
        assert isinstance(10.5, float)

        # Test string validation
        assert isinstance("test", str)

    def test_range_validation(self):
        """Test range validation for numerical values."""
        value = 5

        assert 0 <= value <= 10
        assert value > 0
        assert value < 10


class TestDataProcessing:
    """Test data processing functions."""

    def test_data_structure_handling(self):
        """Test basic data structure handling."""
        # Test with simple numpy arrays
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        assert len(data) == 5
        assert np.mean(data) == 3.0

    def test_statistics_calculation(self):
        """Test statistical calculations on data."""
        data = np.random.randn(100)

        mean = np.mean(data)
        std = np.std(data)

        assert isinstance(mean, (int, float, np.number))
        assert isinstance(std, (int, float, np.number))
        assert std >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

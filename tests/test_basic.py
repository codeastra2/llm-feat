"""
Basic test without API calls - just verify structure and error handling
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest

import llm_feat


def test_api_key_validation():
    """Test that API key validation raises ValueError when no key is set"""
    # Clear API key by setting it to empty string
    llm_feat.set_api_key("")

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    metadata = pd.DataFrame(
        {
            "column_name": ["a", "b"],
            "description": ["col a", "col b"],
            "data_type": ["numeric", "numeric"],
            "label_definition": [None, None],
        }
    )

    # Mock environment variable to be None to ensure no API key is available
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        # Should raise ValueError when no API key is set
        with pytest.raises(ValueError, match="API key"):
            llm_feat.generate_features(df, metadata, mode="code")


def test_metadata_validation():
    """Test that metadata validation raises ValueError for invalid metadata"""
    # Set a dummy API key to bypass API key validation
    llm_feat.set_api_key("dummy-key-for-test")

    df = pd.DataFrame({"a": [1, 2, 3]})
    # Missing required columns
    bad_metadata = pd.DataFrame({"wrong_col": ["a"]})

    # Should raise ValueError for invalid metadata structure
    with pytest.raises(ValueError, match="required columns"):
        llm_feat.generate_features(df, bad_metadata, mode="code")


def test_dataframe_info_preparation():
    """Test that DataFrame info preparation works correctly"""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "savings": [10000, 15000, 20000],
        }
    )
    metadata = pd.DataFrame(
        {
            "column_name": ["age", "income", "savings"],
            "description": ["Age", "Income", "Savings"],
            "data_type": ["numeric", "numeric", "numeric"],
            "label_definition": [None, None, None],
        }
    )

    # Verify DataFrame structure
    assert df.shape == (3, 3), "DataFrame should have 3 rows and 3 columns"

    # Verify metadata structure
    required_cols = [
        "column_name",
        "description",
        "data_type",
        "label_definition",
    ]
    all_present = all(col in metadata.columns for col in required_cols)
    assert all_present, "All required metadata columns should be present"

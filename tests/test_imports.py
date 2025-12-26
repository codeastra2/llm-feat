"""
Quick test to verify package structure and imports
"""

import llm_feat


def test_imports():
    """Test that all imports work correctly"""
    # Check main functions are available
    assert hasattr(llm_feat, "set_api_key"), "set_api_key not found"
    assert hasattr(llm_feat, "generate_features"), "generate_features not found"


def test_version():
    """Test that version is accessible"""
    assert hasattr(llm_feat, "__version__"), "__version__ not found"
    version = llm_feat.__version__
    assert isinstance(version, str), "Version should be a string"
    assert len(version) > 0, "Version should not be empty"

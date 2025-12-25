"""
Quick test to verify package structure and imports
"""

def test_imports():
    """Test that all imports work correctly"""
    try:
        import llm_feat
        print("✓ Successfully imported llm_feat")
        
        # Check main functions are available
        assert hasattr(llm_feat, 'set_api_key'), "set_api_key not found"
        assert hasattr(llm_feat, 'generate_features'), "generate_features not found"
        print("✓ All main functions are available")
        
        # Check version
        print(f"✓ Package version: {llm_feat.__version__}")
        
        print("\n✅ All imports successful! Package structure is correct.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise
    except AssertionError as e:
        print(f"❌ {e}")
        raise

if __name__ == "__main__":
    test_imports()


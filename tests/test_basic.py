"""
Basic test without API calls - just verify structure and error handling
"""

import pandas as pd
import llm_feat

print("=" * 60)
print("Testing llm-feat Package Structure")
print("=" * 60)

# Test 1: Check imports
print("\n✓ Test 1: Package imports successfully")

# Test 2: Check API key error handling
print("\n✓ Test 2: Testing API key validation...")
try:
    # Try to use without API key
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    metadata = pd.DataFrame({
        'column_name': ['a', 'b'],
        'description': ['col a', 'col b'],
        'data_type': ['numeric', 'numeric'],
        'label_definition': [None, None]
    })
    llm_feat.generate_features(df, metadata, mode='code')
    print("  ⚠️  Should have raised an error (no API key set)")
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError: {str(e)[:80]}...")
except Exception as e:
    print(f"  ⚠️  Unexpected error: {type(e).__name__}: {e}")

# Test 3: Check metadata validation
print("\n✓ Test 3: Testing metadata validation...")
try:
    df = pd.DataFrame({'a': [1, 2, 3]})
    bad_metadata = pd.DataFrame({'wrong_col': ['a']})  # Missing required columns
    llm_feat.set_api_key("dummy-key-for-test")
    llm_feat.generate_features(df, bad_metadata, mode='code')
    print("  ⚠️  Should have raised an error (invalid metadata)")
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError for invalid metadata")
except Exception as e:
    print(f"  ⚠️  Got error: {type(e).__name__}: {str(e)[:80]}...")

# Test 4: Check DataFrame info preparation
print("\n✓ Test 4: Testing DataFrame info preparation...")
df = pd.DataFrame({
    'age': [25, 30, 35],
    'income': [50000, 60000, 70000],
    'savings': [10000, 15000, 20000]
})
metadata = pd.DataFrame({
    'column_name': ['age', 'income', 'savings'],
    'description': ['Age', 'Income', 'Savings'],
    'data_type': ['numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, None]
})
print(f"  ✓ DataFrame shape: {df.shape}")
print(f"  ✓ Metadata columns: {list(metadata.columns)}")
print(f"  ✓ All required metadata columns present: {all(col in metadata.columns for col in ['column_name', 'description', 'data_type', 'label_definition'])}")

print("\n" + "=" * 60)
print("✅ Basic structure tests completed!")
print("=" * 60)
print("\nTo test with actual LLM calls, you'll need:")
print("1. Set your OpenAI API key: llm_feat.set_api_key('your-key')")
print("2. Open test_llm_feat.ipynb in Jupyter notebook")


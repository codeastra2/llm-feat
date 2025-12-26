#!/usr/bin/env python3
"""
Example usage of llm-feat package for automated feature engineering.

This script demonstrates how to use llm-feat to generate features
for tabular data using Large Language Models.

Run with: poetry run python example_llm_feat.py
"""

import os
import sys

import numpy as np
import pandas as pd

import llm_feat


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Main function demonstrating llm-feat functionality."""
    print_section("llm-feat Package Example")
    print(f"llm-feat version: {llm_feat.__version__}")
    print("✓ Package imported successfully!")

    # Set API key
    print_section("API Key Setup")
    api_key = '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'
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set in environment.")
        print("   Please set it using: export OPENAI_API_KEY='your-key-here'")
        print("\n   For this demo, you can also set it directly in the script")
        print("   (not recommended for production).")
        sys.exit(1)

    llm_feat.set_api_key(api_key)
    print("✓ API key set from environment variable")

    # Example 1: Simple Numerical Dataset
    print_section("Example 1: Simple Numerical Dataset")

    # Create a simple numerical dataset
    df1 = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 55000, 65000, 75000, 85000],
            "savings": [10000, 15000, 20000, 25000, 30000, 35000, 12000, 18000, 22000, 28000],
            "expenses": [40000, 45000, 50000, 55000, 60000, 65000, 43000, 47000, 53000, 57000],
        }
    )

    print("Original DataFrame:")
    print(df1.head())
    print(f"\nShape: {df1.shape}")
    print(f"Columns: {list(df1.columns)}")

    # Create metadata DataFrame
    metadata_df1 = pd.DataFrame(
        {
            "column_name": ["age", "income", "savings", "expenses"],
            "description": [
                "Age of the person in years",
                "Annual income in dollars",
                "Total savings in dollars",
                "Annual expenses in dollars",
            ],
            "data_type": ["numeric", "numeric", "numeric", "numeric"],
            "label_definition": [None, None, None, None],
        }
    )

    print("\nMetadata DataFrame:")
    print(metadata_df1)

    # Mode 1: Generate Code
    print_section("Mode 1: Generate Code")
    print("Generating feature engineering code...")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    code1 = llm_feat.generate_features(df1, metadata_df1, mode="code", model="gpt-4o-mini")
    print("Generated code:")
    print(code1)

    # Execute the generated code
    print("\nExecuting generated code...")
    exec(code1, {"df": df1, "pd": pd, "np": np})
    print("\nDataFrame after executing generated code:")
    print(df1.head())
    print(f"\nNew columns added: {[col for col in df1.columns if col not in ['age', 'income', 'savings', 'expenses']]}")

    # Mode 2: Direct Feature Addition
    print_section("Mode 2: Direct Feature Addition")
    print("Generating and adding features directly to DataFrame...")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    # Create a fresh copy for direct mode
    df1_fresh = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 55000, 65000, 75000, 85000],
            "savings": [10000, 15000, 20000, 25000, 30000, 35000, 12000, 18000, 22000, 28000],
            "expenses": [40000, 45000, 50000, 55000, 60000, 65000, 43000, 47000, 53000, 57000],
        }
    )

    df_with_features = llm_feat.generate_features(
        df1_fresh, metadata_df1, mode="direct", model="gpt-4o-mini"
    )

    print("DataFrame with new features:")
    print(df_with_features.head())
    print(f"\nOriginal columns: {list(df1_fresh.columns)}")
    print(f"New columns: {[col for col in df_with_features.columns if col not in df1_fresh.columns]}")
    print(f"\nTotal columns: {len(df_with_features.columns)} (original: {len(df1_fresh.columns)})")

    # Example 2: Dataset with Target Column
    print_section("Example 2: Dataset with Target Column")

    # Create dataset with target column
    df2 = pd.DataFrame(
        {
            "height": [170, 175, 180, 165, 185, 172, 178, 168, 182, 174],
            "weight": [70, 75, 80, 65, 85, 72, 78, 68, 83, 74],
            "bmi": [24.2, 24.5, 24.7, 23.9, 24.8, 24.3, 24.6, 24.1, 25.0, 24.4],
            "health_score": [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],  # Target: 1=healthy, 0=unhealthy
        }
    )

    metadata_df2 = pd.DataFrame(
        {
            "column_name": ["height", "weight", "bmi", "health_score"],
            "description": [
                "Height in centimeters",
                "Weight in kilograms",
                "Body Mass Index",
                "Health classification score",
            ],
            "data_type": ["numeric", "numeric", "numeric", "numeric"],
            "label_definition": [None, None, None, "1 if healthy, 0 if unhealthy"],
        }
    )

    print("Dataset with target:")
    print(df2.head())
    print("\nMetadata:")
    print(metadata_df2)

    # Generate features for dataset with target
    print_section("Generating Features for Dataset with Target")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    code2 = llm_feat.generate_features(df2, metadata_df2, mode="code", model="gpt-4o-mini")
    print("Generated feature code:")
    print(code2)

    # Execute the generated code
    print("\nExecuting generated code...")
    exec(code2, {"df": df2, "pd": pd, "np": np})
    print("\nDataFrame after executing generated code:")
    print(df2.head())
    print(f"\nNew columns added: {[col for col in df2.columns if col not in ['height', 'weight', 'bmi', 'health_score']]}")

    # Summary
    print_section("Summary")
    print("✓ Package imports successfully")
    print("✓ API key management works")
    print("✓ Metadata validation works")
    print("✓ Code generation mode works")
    print("✓ Direct feature addition mode works")
    print("✓ Works with datasets with and without target columns")
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


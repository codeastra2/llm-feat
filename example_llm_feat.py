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
    api_key = os.getenv("OPENAI_API_KEY")
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
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 55000, 65000, 75000, 85000],
            "savings": [10000, 15000, 20000, 25000, 30000, 35000, 12000, 18000, 22000, 28000],
            "expenses": [40000, 45000, 50000, 55000, 60000, 65000, 43000, 47000, 53000, 57000],
        }
    )

    print("Original DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

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

    code1 = llm_feat.generate_features(df, metadata_df1, mode="code", model="gpt-4o-mini")
    print("Generated code:")
    print(code1)

    # Execute the generated code
    print("\nExecuting generated code...")
    exec(code1, {"df": df, "pd": pd, "np": np})
    print("\nDataFrame after executing generated code:")
    print(df.head())
    print(
        f"\nNew columns added: {[col for col in df.columns if col not in ['age', 'income', 'savings', 'expenses']]}"
    )

    # Mode 2: Direct Feature Addition
    print_section("Mode 2: Direct Feature Addition")
    print("Generating and adding features directly to DataFrame...")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    # Create a fresh copy for direct mode
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 28, 32, 38, 42],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 55000, 65000, 75000, 85000],
            "savings": [10000, 15000, 20000, 25000, 30000, 35000, 12000, 18000, 22000, 28000],
            "expenses": [40000, 45000, 50000, 55000, 60000, 65000, 43000, 47000, 53000, 57000],
        }
    )

    df = llm_feat.generate_features(df, metadata_df1, mode="direct", model="gpt-4o-mini")

    print("DataFrame with new features:")
    print(df.head())
    original_cols = ["age", "income", "savings", "expenses"]
    print(f"\nOriginal columns: {original_cols}")
    print(f"New columns: {[col for col in df.columns if col not in original_cols]}")
    print(f"\nTotal columns: {len(df.columns)} (original: {len(original_cols)})")

    # Example 2: Dataset with Target Column
    print_section("Example 2: Dataset with Target Column")

    # Create dataset with target column
    df = pd.DataFrame(
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
    print(df.head())
    print("\nMetadata:")
    print(metadata_df2)

    # Generate features for dataset with target
    print_section("Generating Features for Dataset with Target")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    code2 = llm_feat.generate_features(df, metadata_df2, mode="code", model="gpt-4o-mini")
    print("Generated feature code:")
    print(code2)

    # Execute the generated code
    print("\nExecuting generated code...")
    exec(code2, {"df": df, "pd": pd, "np": np})
    print("\nDataFrame after executing generated code:")
    print(df.head())
    print(
        f"\nNew columns added: {[col for col in df.columns if col not in ['height', 'weight', 'bmi', 'health_score']]}"
    )

    # Example with Feature Report
    print("\n" + "-" * 70)
    print("Generating Features with Report")
    print("-" * 70 + "\n")

    code2_with_report, report2 = llm_feat.generate_features(
        df, metadata_df2, mode="code", model="gpt-4o-mini", return_report=True
    )
    print("Generated feature code:")
    print(code2_with_report)
    print("\n" + "=" * 70)
    print("FEATURE REPORT")
    print("=" * 70)
    print(report2)

    # Example 3: Partial Metadata (Many Columns, Few Descriptions)
    print_section("Example 3: Partial Metadata (Many Columns, Few Descriptions)")
    print(
        "This example demonstrates that you can provide metadata for only a subset of columns. "
        "The LLM will still see all columns in the DataFrame and can generate features using all of them, "
        "but will have richer context for columns with descriptions.\n"
    )

    # Create a dataset with many columns (simulating a real-world scenario)
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, 30, 35, 40, 45, 28, 32, 38, 42, 50],
            "income": [50000, 60000, 70000, 80000, 90000, 55000, 65000, 75000, 85000, 95000],
            "credit_score": [650, 720, 680, 750, 710, 660, 730, 690, 740, 700],
            "loan_amount": [10000, 15000, 20000, 25000, 30000, 12000, 18000, 22000, 28000, 35000],
            "employment_years": [2, 5, 8, 12, 15, 3, 6, 9, 11, 18],
            "debt_to_income": [0.25, 0.30, 0.28, 0.35, 0.32, 0.27, 0.29, 0.31, 0.33, 0.36],
            "savings": [5000, 8000, 12000, 15000, 20000, 6000, 9000, 13000, 16000, 22000],
            "num_accounts": [2, 3, 2, 4, 3, 2, 3, 2, 4, 3],
            "default_status": [
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
            ],  # Target: 1=defaulted, 0=not defaulted
        }
    )

    print("DataFrame with many columns:")
    print(df.head())
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Provide metadata for only a FEW important columns (not all columns)
    metadata_df3 = pd.DataFrame(
        {
            "column_name": ["income", "credit_score", "loan_amount", "default_status"],
            "description": [
                "Annual income in dollars",
                "Credit score (300-850 scale)",
                "Loan amount requested in dollars",
                "Loan default status",
            ],
            "data_type": ["numeric", "numeric", "numeric", "numeric"],
            "label_definition": [None, None, None, "1 if customer defaulted on loan, 0 if not"],
        }
    )

    print("\nMetadata provided for only 4 out of 10 columns:")
    print(metadata_df3)
    print(f"\nColumns with metadata: {list(metadata_df3['column_name'])}")
    print(
        f"Columns without metadata: {[col for col in df.columns if col not in metadata_df3['column_name'].values]}"
    )

    print("\nGenerating features - LLM will use ALL columns, not just the ones with metadata...")
    print("(Using gpt-4o-mini model for cost-effective feature generation)\n")

    code3 = llm_feat.generate_features(df, metadata_df3, mode="code", model="gpt-4o-mini")
    print("Generated feature code (using all columns, with context from described columns):")
    print(code3)

    # Direct feature addition with partial metadata
    print("\n" + "-" * 70)
    print("Direct Feature Addition with Partial Metadata")
    print("-" * 70 + "\n")

    df = llm_feat.generate_features(df, metadata_df3, mode="direct", model="gpt-4o-mini")

    print("DataFrame with new features:")
    print(df.head())
    original_cols_count = 10
    print(f"\nOriginal columns: {original_cols_count}")
    print(f"New columns added: {len(df.columns) - original_cols_count}")
    print(f"Total columns: {len(df.columns)}")

    print(
        "\n**Key Takeaway:** You don't need to provide metadata for every column! "
        "Provide descriptions for the target column, important domain-specific columns, "
        "and columns that need special handling. The LLM will still see all columns in your "
        "DataFrame and can generate features using all of them."
    )

    # Example 4: Using Problem Description for Additional Context
    print_section("Example 4: Using Problem Description for Additional Context")
    print(
        "This example demonstrates how to use the `problem_description` parameter to provide "
        "additional business context to the LLM, which helps generate more relevant and "
        "domain-specific features.\n"
    )

    # Create a dataset for e-commerce customer churn prediction
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "total_purchases": [15, 8, 25, 3, 30, 12, 20, 5, 18, 22],
            "avg_order_value": [
                45.50,
                32.00,
                78.90,
                15.00,
                95.20,
                55.30,
                67.80,
                20.50,
                72.40,
                88.60,
            ],
            "days_since_last_purchase": [5, 45, 2, 90, 1, 30, 7, 120, 10, 3],
            "support_tickets": [0, 2, 0, 5, 1, 1, 0, 3, 0, 0],
            "account_age_days": [365, 180, 730, 90, 1095, 240, 540, 60, 450, 600],
            "churned": [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # Target: 1=churned, 0=active
        }
    )

    metadata_df4 = pd.DataFrame(
        {
            "column_name": [
                "total_purchases",
                "avg_order_value",
                "days_since_last_purchase",
                "support_tickets",
                "account_age_days",
                "churned",
            ],
            "description": [
                "Total number of purchases made by customer",
                "Average value of customer orders in dollars",
                "Number of days since customer last made a purchase",
                "Number of customer support tickets opened",
                "Age of customer account in days",
                "Customer churn status",
            ],
            "data_type": ["numeric", "numeric", "numeric", "numeric", "numeric", "numeric"],
            "label_definition": [
                None,
                None,
                None,
                None,
                None,
                "1 if customer churned, 0 if active",
            ],
        }
    )

    print("E-commerce Customer Dataset:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    # Without Problem Description
    print("\n" + "-" * 70)
    print("Without Problem Description")
    print("-" * 70 + "\n")

    code4_no_desc = llm_feat.generate_features(df, metadata_df4, mode="code", model="gpt-4o-mini")
    print("Generated code WITHOUT problem description:")
    print(code4_no_desc)

    # With Problem Description
    print("\n" + "-" * 70)
    print("With Problem Description")
    print("-" * 70 + "\n")

    # Define the problem description with business context
    problem_desc = """We are an e-commerce company trying to predict customer churn.
Key business insights:
- Customers who haven't purchased in 30+ days are at high risk of churning
- High support ticket volume often indicates dissatisfaction and leads to churn
- Customers with high lifetime value (many purchases, high order values) are less likely to churn
- New customers (account age < 90 days) with low engagement are particularly vulnerable
- We want to identify at-risk customers early to intervene with retention campaigns

Generate features that help identify these at-risk customer segments.
"""

    # Generate features WITH problem description
    code4_with_desc = llm_feat.generate_features(
        df, metadata_df4, mode="code", model="gpt-4o-mini", problem_description=problem_desc
    )
    print("Generated code WITH problem description:")
    print(code4_with_desc)

    # Direct Mode with Problem Description
    print("\n" + "-" * 70)
    print("Direct Mode with Problem Description")
    print("-" * 70 + "\n")

    df = llm_feat.generate_features(
        df, metadata_df4, mode="direct", model="gpt-4o-mini", problem_description=problem_desc
    )

    print("DataFrame with features generated using problem description:")
    print(df.head())
    original_cols_count = 7
    print(f"\nOriginal columns: {original_cols_count}")
    print(f"New columns added: {len(df.columns) - original_cols_count}")
    print(f"Total columns: {len(df.columns)}")
    original_cols_list = [
        "customer_id",
        "total_purchases",
        "avg_order_value",
        "days_since_last_purchase",
        "support_tickets",
        "account_age_days",
        "churned",
    ]
    print(f"\nNew feature columns: {[col for col in df.columns if col not in original_cols_list]}")

    print(
        "\n**Key Takeaway:** The `problem_description` parameter allows you to provide business context, "
        "domain knowledge, and specific requirements to the LLM. This helps generate features that are "
        "more aligned with your business objectives, better suited for your specific use case, and "
        "tailored to domain-specific patterns and insights."
    )

    # Example 5: Feature Report
    print_section("Example 5: Generating Feature Report")
    print(
        "This example demonstrates how to get a detailed report explaining the domain understanding "
        "and rationale for each generated feature.\n"
    )

    # Use a simple dataset for report demonstration
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "credit_score": [650, 720, 680, 750, 710],
            "loan_default": [0, 0, 1, 0, 1],  # Target
        }
    )

    metadata_df5 = pd.DataFrame(
        {
            "column_name": ["age", "income", "credit_score", "loan_default"],
            "description": [
                "Age of the borrower",
                "Annual income in dollars",
                "Credit score (300-850)",
                "Loan default status",
            ],
            "data_type": ["numeric", "numeric", "numeric", "numeric"],
            "label_definition": [None, None, None, "1 if defaulted, 0 if not"],
        }
    )

    print("Dataset:")
    print(df.head())

    # Generate with report
    code5, report5 = llm_feat.generate_features(
        df, metadata_df5, mode="code", model="gpt-4o-mini", return_report=True
    )

    print("\nGenerated Code:")
    print(code5)
    print("\n" + "=" * 70)
    print("FEATURE REPORT")
    print("=" * 70)
    print(report5)

    # Direct mode with report
    print("\n" + "-" * 70)
    print("Direct Mode with Report")
    print("-" * 70 + "\n")

    # Create a fresh copy for direct mode
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "credit_score": [650, 720, 680, 750, 710],
            "loan_default": [0, 0, 1, 0, 1],  # Target
        }
    )

    df, report5_direct = llm_feat.generate_features(
        df, metadata_df5, mode="direct", model="gpt-4o-mini", return_report=True
    )

    print("DataFrame with features:")
    print(df.head())
    print("\n" + "=" * 70)
    print("FEATURE REPORT")
    print("=" * 70)
    print(report5_direct)

    # Summary
    print_section("Summary")
    print("✓ Package imports successfully")
    print("✓ API key management works")
    print("✓ Metadata validation works")
    print("✓ Code generation mode works")
    print("✓ Direct feature addition mode works")
    print("✓ Works with datasets with and without target columns")
    print(
        "✓ Works with partial metadata (you can provide descriptions for only a subset of columns)"
    )
    print("✓ Supports problem description for additional business context")
    print("✓ Can generate feature reports with domain understanding and feature explanations")
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

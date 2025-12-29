"""Core functionality for llm-feat"""

from typing import Literal, Optional

import pandas as pd

from .jupyter_utils import get_code_string, inject_code_to_next_cell, is_jupyter
from .llm_client import LLMClient

# Global API key storage
_API_KEY: Optional[str] = None
_LLM_CLIENT: Optional[LLMClient] = None


def set_api_key(api_key: str) -> None:
    """
    Set the OpenAI API key for the session.

    Args:
        api_key: Your OpenAI API key
    """
    global _API_KEY, _LLM_CLIENT
    _API_KEY = api_key
    _LLM_CLIENT = None  # Reset client to use new key


def _get_client() -> LLMClient:
    """Get or create LLM client instance"""
    global _LLM_CLIENT, _API_KEY

    if _LLM_CLIENT is None:
        _LLM_CLIENT = LLMClient(api_key=_API_KEY)

    return _LLM_CLIENT


def _prepare_df_info(df: pd.DataFrame, metadata_df: Optional[pd.DataFrame] = None) -> str:
    """Prepare DataFrame information string for LLM"""
    info_lines = [
        f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
        "\nColumns and Data Types:",
    ]

    for col in df.columns:
        dtype = str(df[col].dtype)
        info_lines.append(f"  - {col}: {dtype}")

    # Add sample data for numerical columns
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numerical_cols:
        info_lines.append("\nSample data (first 3 rows) for numerical columns:")
        sample_df = df[numerical_cols].head(3)
        info_lines.append(sample_df.to_string())

    # Add basic statistics
    if numerical_cols:
        info_lines.append("\nBasic statistics for numerical columns:")
        stats = df[numerical_cols].describe()
        info_lines.append(stats.to_string())

    # Add categorical column information if metadata is provided
    if metadata_df is not None:
        categorical_cols = _get_categorical_columns(df, metadata_df)
        if categorical_cols:
            info_lines.append("\nCategorical Columns Information:")
            for col in categorical_cols:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    unique_count = len(unique_vals)
                    info_lines.append(f"\n  Column: {col}")
                    info_lines.append(f"    Unique values count: {unique_count}")
                    if unique_count <= 20:  # Show all unique values if <= 20
                        info_lines.append(f"    Unique values: {list(unique_vals)}")
                    else:  # Show sample if too many
                        sample_vals = list(unique_vals[:10])
                        info_lines.append(f"    Sample unique values (first 10): " f"{sample_vals}")
                        info_lines.append(f"    ... and {unique_count - 10} more " f"unique values")
                    # Add value counts for top categories
                    value_counts = df[col].value_counts().head(5)
                    info_lines.append("    Top 5 value counts:")
                    for val, count in value_counts.items():
                        pct = count / len(df) * 100
                        info_lines.append(f"      '{val}': {count} ({pct:.1f}%)")

    return "\n".join(info_lines)


def _prepare_metadata_info(metadata_df: pd.DataFrame) -> str:
    """Prepare metadata DataFrame information string for LLM"""
    required_cols = [
        "column_name",
        "description",
        "data_type",
        "label_definition",
    ]

    # Validate metadata structure
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata DataFrame missing required columns: "
            f"{missing_cols}. Required columns: {required_cols}"
        )

    info_lines = ["Column Metadata:"]
    for _, row in metadata_df.iterrows():
        col_name = row["column_name"]
        description = row["description"] if pd.notna(row["description"]) else "N/A"
        data_type = row["data_type"] if pd.notna(row["data_type"]) else "N/A"
        label_def = row["label_definition"] if pd.notna(row["label_definition"]) else "N/A"

        info_lines.append(f"\n  Column: {col_name}")
        info_lines.append(f"    Description: {description}")
        info_lines.append(f"    Data Type: {data_type}")
        if label_def != "N/A":
            info_lines.append(f"    Label Definition: {label_def}")

    return "\n".join(info_lines)


def _extract_target_column(metadata_df: pd.DataFrame) -> Optional[str]:
    """Extract target/label column name from metadata"""
    if "label_definition" in metadata_df.columns:
        label_rows = metadata_df[
            metadata_df["label_definition"].notna() & (metadata_df["label_definition"] != "")
        ]
        if not label_rows.empty:
            return label_rows.iloc[0]["column_name"]
    return None


def _get_categorical_columns(df: pd.DataFrame, metadata_df: pd.DataFrame) -> list:
    """Extract list of categorical column names from metadata"""
    categorical_cols = []
    if "data_type" in metadata_df.columns and "column_name" in metadata_df.columns:
        cat_rows = metadata_df[
            metadata_df["data_type"]
            .str.lower()
            .isin(
                [
                    "categorical",
                    "category",
                    "cat",
                    "string",
                    "text",
                    "object",
                ]
            )
        ]
        categorical_cols = [
            row["column_name"] for _, row in cat_rows.iterrows() if row["column_name"] in df.columns
        ]
    return categorical_cols


def generate_features(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    mode: Literal["code", "direct"] = "code",
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    debug: bool = False,
    problem_description: Optional[str] = None,
    return_report: bool = False,
) -> pd.DataFrame | str | tuple[str, str] | tuple[pd.DataFrame, str]:
    """
    Generate feature engineering code or directly add features to DataFrame.

    Args:
        df: Input pandas DataFrame
        metadata_df: Metadata DataFrame with columns: column_name,
                    description, data_type, label_definition. Set data_type
                    to 'categorical' for categorical columns to enable
                    categorical feature engineering.
        mode: 'code' to return/suggest code, 'direct' to add features
              directly
        api_key: OpenAI API key (optional if already set via
                set_api_key())
        model: OpenAI model to use (default: "gpt-4o", alternatives:
               "gpt-4-turbo", "gpt-3.5-turbo")
        debug: If True, print the generated code before execution (useful
               for troubleshooting)
        problem_description: Optional description of the problem/use case
                           to provide additional context to the LLM for
                           generating more relevant features
        return_report: If True, also return a feature report containing
                      domain understanding and explanations for each generated
                      feature

    Note:
        Generated code uses 'df' as the DataFrame variable name.
        If your DataFrame has a different name, replace 'df' with your
        variable name in the generated code.

    Returns:
        - If mode='code' and return_report=False: Returns code string (and injects to next cell
          if in Jupyter)
        - If mode='code' and return_report=True: Returns tuple (code, report) where report
          contains domain understanding and feature explanations
        - If mode='direct' and return_report=False: Returns DataFrame with new features added
        - If mode='direct' and return_report=True: Returns tuple (DataFrame, report)

    Note:
        Categorical features are automatically detected from metadata_df
        where data_type is 'categorical', 'category', 'cat', 'string',
        'text', or 'object'. The LLM will generate appropriate encoding
        strategies (one-hot, target encoding, frequency encoding, etc.)
        based on the unique value counts.
    """
    # Set API key if provided
    if api_key:
        set_api_key(api_key)

    # Prepare information for LLM
    df_info = _prepare_df_info(df, metadata_df)
    metadata_info = _prepare_metadata_info(metadata_df)
    target_column = _extract_target_column(metadata_df)
    categorical_cols = _get_categorical_columns(df, metadata_df)

    # Generate feature code using LLM
    client = _get_client()
    result = client.generate_feature_code(
        df_info,
        metadata_info,
        target_column,
        categorical_cols,
        model=model,
        problem_description=problem_description,
        return_report=return_report,
    )

    if return_report:
        generated_code, feature_report = result
    else:
        generated_code = result
        feature_report = None

    # Validate that generated code contains DataFrame assignments
    if mode == "direct":
        # Check if code contains df['...'] = patterns
        has_df_assignments = (
            "df['" in generated_code
            or 'df["' in generated_code
            or "df.loc" in generated_code
            or "pd.get_dummies" in generated_code
        )
        if not has_df_assignments:
            import warnings

            warnings.warn(
                "Generated code may not create new columns. "
                "Code should contain patterns like df['new_col'] = ... "
                f"Generated code:\n{generated_code[:500]}"
            )

    # Debug: print generated code if requested
    if debug:
        print("=" * 60)
        print("GENERATED CODE:")
        print("=" * 60)
        print(generated_code)
        print("=" * 60)

    if mode == "code":
        # Code generation mode
        if is_jupyter():
            # Try to inject into next cell
            inject_code_to_next_cell(generated_code)

        # Also return as string
        code_string = get_code_string(generated_code)
        if return_report:
            return code_string, feature_report
        return code_string

    elif mode == "direct":
        # Direct feature addition mode
        # Create a copy to avoid modifying original
        df_result = df.copy()

        # Store original column count for validation
        original_cols = set(df_result.columns)
        original_col_count = len(df_result.columns)

        # Execute the generated code in a safe context
        # We'll use exec with restricted globals
        import numpy as np

        safe_globals = {
            "df": df_result,
            "pd": pd,
            "np": np,
        }

        try:
            # Execute the generated code
            # Note: The code should modify 'df' in place
            # (e.g., df['new_col'] = ...)
            exec(generated_code, safe_globals)

            # After execution, get the DataFrame from safe_globals
            # This ensures we get the modified version even if code
            # reassigned df
            if "df" in safe_globals:
                df_result = safe_globals["df"]

            # Verify we still have a DataFrame
            if not isinstance(df_result, pd.DataFrame):
                raise RuntimeError(
                    "After code execution, 'df' is not a DataFrame. "
                    f"Type: {type(df_result)}. "
                    f"Generated code:\n{generated_code}"
                )

            # Check if new columns were actually added
            new_cols = set(df_result.columns) - original_cols
            new_col_count = len(df_result.columns) - original_col_count

            if not new_cols and new_col_count == 0:
                # No new columns were added - this might indicate the
                # code didn't work. This could happen if:
                # 1. The LLM generated code that doesn't create features
                # 2. The code has errors that were silently ignored
                # 3. The code creates features but with existing column
                #    names
                import warnings

                warnings.warn(
                    "No new columns were added after executing generated "
                    "code. "
                    f"Original columns: {original_col_count}, "
                    f"After execution: {len(df_result.columns)}. "
                    "This might indicate the generated code didn't create "
                    "features. "
                    "Try using mode='code' to review the generated code "
                    "first.\n"
                    f"Generated code:\n{generated_code}",
                    UserWarning,
                )

            if return_report:
                return df_result, feature_report
            return df_result
        except Exception as e:
            raise RuntimeError(
                f"Error executing generated feature code: {str(e)}\n"
                f"Generated code:\n{generated_code}"
            )

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'code' or 'direct'")

# API Reference

Complete documentation for all public functions and parameters in llm-feat.

## Core Functions

### `set_api_key(api_key: str) -> None`

Set the OpenAI API key for the current session.

**Parameters:**
- `api_key` (str): Your OpenAI API key

**Example:**
```python
import llm_feat
llm_feat.set_api_key("sk-...")
```

**Note:** You can also set the `OPENAI_API_KEY` environment variable instead.

---

### `generate_features(df, metadata_df, mode='code', api_key=None, model='gpt-4o', debug=False, problem_description=None, return_report=False)`

Generate feature engineering code or directly add features to your DataFrame.

**Parameters:**

- **df** (`pd.DataFrame`): Input pandas DataFrame containing your data
- **metadata_df** (`pd.DataFrame`): Metadata DataFrame with the following required columns:
  - `column_name`: Name of each column
  - `description`: Human-readable description of what the column represents
  - `data_type`: Data type (`'numeric'`, `'categorical'`, `'category'`, `'cat'`, `'string'`, `'text'`, or `'object'`)
  - `label_definition`: Definition of the target variable (if applicable). Set to `None` for non-target columns.
- **mode** (`'code'` | `'direct'`, default: `'code'`): 
  - `'code'`: Returns Python code string (recommended for Jupyter notebooks)
  - `'direct'`: Executes code and returns DataFrame with new features added
- **api_key** (`str`, optional): OpenAI API key. If not provided, uses the key set via `set_api_key()` or `OPENAI_API_KEY` environment variable.
- **model** (`str`, default: `'gpt-4o'`): OpenAI model to use. Options: `'gpt-4o'`, `'gpt-4-turbo'`, `'gpt-4o-mini'`, `'gpt-3.5-turbo'`
- **debug** (`bool`, default: `False`): If `True`, prints the generated code before execution (useful for troubleshooting)
- **problem_description** (`str`, optional): Additional context about your problem/use case to help the LLM generate more relevant features
- **return_report** (`bool`, default: `False`): If `True`, returns a feature report containing domain understanding and explanations for each generated feature

**Returns:**

The return type depends on `mode` and `return_report`:

- `mode='code'`, `return_report=False`: Returns `str` (code string)
- `mode='code'`, `return_report=True`: Returns `tuple[str, str]` (code, report)
- `mode='direct'`, `return_report=False`: Returns `pd.DataFrame` (DataFrame with new features)
- `mode='direct'`, `return_report=True`: Returns `tuple[pd.DataFrame, str]` (DataFrame, report)

**Important Notes:**

1. **DataFrame Variable Name**: Generated code uses `df` as the DataFrame variable name. If your DataFrame has a different name, replace `df` with your variable name in the generated code, or rename your DataFrame to `df` before using `mode='direct'`.

2. **Categorical Features**: Set `data_type` to `'categorical'`, `'category'`, `'cat'`, `'string'`, `'text'`, or `'object'` in `metadata_df` to enable categorical feature engineering. The LLM will automatically choose appropriate encoding strategies (one-hot, target encoding, frequency encoding, etc.) based on unique value counts.

3. **Jupyter Integration**: When using `mode='code'` in Jupyter notebooks, the generated code is automatically injected into the next cell.

**Example 1: Get Code (Recommended)**
```python
import pandas as pd
import llm_feat

df = pd.DataFrame({
    'income': [50000, 60000, 70000],
    'expenses': [30000, 35000, 40000],
    'target': [1, 0, 1]
})

metadata_df = pd.DataFrame({
    'column_name': ['income', 'expenses', 'target'],
    'description': ['Annual income', 'Annual expenses', 'Binary target'],
    'data_type': ['numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, '1 if positive, 0 if negative']
})

code = llm_feat.generate_features(df, metadata_df, mode='code')
print(code)
```

**Example 2: Direct Feature Addition**
```python
df_with_features = llm_feat.generate_features(
    df, metadata_df, mode='direct', model='gpt-4o-mini'
)
print(df_with_features.head())
```

**Example 3: With Problem Description**
```python
code = llm_feat.generate_features(
    df, 
    metadata_df, 
    mode='code',
    problem_description="Predicting customer churn for a subscription service"
)
```

**Example 4: With Feature Report**
```python
code, report = llm_feat.generate_features(
    df, 
    metadata_df, 
    mode='code',
    return_report=True
)
print(report)
```

**Example 5: Categorical Features**
```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C'],
    'value': [10, 20, 15, 25],
    'target': [1, 0, 1, 0]
})

metadata_df = pd.DataFrame({
    'column_name': ['category', 'value', 'target'],
    'description': ['Product category', 'Sales value', 'Target variable'],
    'data_type': ['categorical', 'numeric', 'numeric'],  # 'categorical' enables encoding
    'label_definition': [None, None, 'Binary classification target']
})

code = llm_feat.generate_features(df, metadata_df, mode='code')
```

---

## Feature Report

When `return_report=True`, the function returns a detailed report containing:

1. **Domain Understanding**: Summary of the problem domain based on metadata and problem description
2. **Generated Features Explanation**: For each feature:
   - Feature name
   - Description of what the feature represents
   - Rationale for why it's useful
   - Domain relevance

The report helps you understand the reasoning behind each generated feature and validate that they make sense for your specific problem.

---

## Version

Access the package version:

```python
import llm_feat
print(llm_feat.__version__)
```


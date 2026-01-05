# llm-feat

[![Python Version](https://img.shields.io/badge/python-3.10.19%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatically generate feature engineering code for pandas DataFrames using LLMs. Get context-aware, target-specific features that understand your domain.

## Installation

```bash
pip install llm-feat
```

## Quick Start

```python
import pandas as pd
import llm_feat

llm_feat.set_api_key("your-openai-api-key")  # or set OPENAI_API_KEY env var

# Your data
df = pd.DataFrame({
    'income': [50000, 60000, 70000],
    'expenses': [30000, 35000, 40000],
    'target': [1, 0, 1]
})

# Metadata describing your columns
metadata_df = pd.DataFrame({
    'column_name': ['income', 'expenses', 'target'],
    'description': ['Annual income', 'Annual expenses', 'Binary target'],
    'data_type': ['numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, '1 if positive, 0 if negative']
})

# Generate features
code = llm_feat.generate_features(df, metadata_df, mode='code')
print(code)
```

**Generated Code:**
```python
import numpy as np

df['income_to_expense_ratio'] = np.where(df['expenses'] != 0, df['income'] / df['expenses'], np.nan)
df['savings'] = df['income'] - df['expenses']
df['savings_to_income_ratio'] = np.where(df['income'] != 0, df['savings'] / df['income'], np.nan)
```

## Feature Reports

Get detailed explanations of why each feature was generated:

```python
code, report = llm_feat.generate_features(
    df, metadata_df, mode='code', return_report=True
)
print(report)
```

**Example Report:**
```
FEATURE REPORT
==============

1. DOMAIN UNDERSTANDING:
   - Problem: Predicting binary target based on income and expenses
   - Key relationships: Income-to-expense ratios indicate financial health

2. GENERATED FEATURES EXPLANATION:
   - Feature: income_to_expense_ratio
     Rationale: Higher ratios indicate better financial stability
     Domain Relevance: Directly related to predicting positive outcomes
```

## Direct Mode

Add features directly to your DataFrame:

```python
df_with_features = llm_feat.generate_features(
    df, metadata_df, mode='direct', model='gpt-4o-mini'
)
```

## Key Features

- **Context-aware**: Uses column descriptions to generate relevant features
- **Target-aware**: Generates features specific to your prediction task
- **Categorical support**: Automatic encoding for categorical columns
- **Jupyter integration**: Code auto-injected into next cell
- **Feature reports**: Understand the reasoning behind each feature

## Documentation

- [Read the Docs](https://llm-feat.readthedocs.io/) - Full documentation
- [API Reference](API.md) - Complete parameter documentation
- [Examples](example_llm_feat.ipynb) - Jupyter notebook examples


## Development

```bash
git clone https://github.com/codeastra2/llm-feat.git
cd llm-feat
conda create -n llm_feat_310 python=3.10.19 -y
conda activate llm_feat_310
poetry install
poetry run pytest
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Srinivas Kumar - [@codeastra2](https://github.com/codeastra2)

## Links

- [GitHub](https://github.com/codeastra2/llm-feat)
- [API Reference](API.md)
- [Changelog](CHANGELOG.md)


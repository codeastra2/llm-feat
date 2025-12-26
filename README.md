# llm-feat

[![Python Version](https://img.shields.io/badge/python-3.10.19%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

llm-feat is a Python library which can  automatically generate feature engineering code in Jupyter notebooks for your pandas datasets using LLM models. The features will be meaningful and can be also automatically added to the pandas dataframe or features addition code can be just modified later to pick and choose features. 

## 🌟 Key Features

- 🤖 **LLM-Powered**: Uses OpenAI models to generate context aware feature engineering code.
- 📝 **Dual Modes**: Get code suggestions or directly add features
- 🔧 **Jupyter Support**: Automatically injects code into next cell in Jupyter notebooks
- 🎯 **Metadata-Driven**: Uses column descriptions to generate contextually relevant features
- 🎯 **Target-Aware**: Generates features specifically relevant to your prediction task

## 📦 Installation

To install the latest release of llm-feat from PyPI:

```bash
pip install llm-feat
```

## 🚀 Quick Start

### 1. Setup

```python
import pandas as pd
import llm_feat

# Set your OpenAI API key
llm_feat.set_api_key("your-openai-api-key-here")
# Or use environment variable: export OPENAI_API_KEY="your-key"
```

### 2. Prepare Data and Metadata

```python
# Your dataset
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'savings': [10000, 15000, 20000, 25000, 30000],
    'expenses': [40000, 45000, 50000, 55000, 60000]
})

# Metadata with column descriptions
metadata_df = pd.DataFrame({
    'column_name': ['age', 'income', 'savings', 'expenses'],
    'description': [
        'Age of the person in years',
        'Annual income in dollars',
        'Total savings in dollars',
        'Annual expenses in dollars'
    ],
    'data_type': ['numeric', 'numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, None, None]
})
```

### 3. Generate Features

#### Mode 1: Get Code (Recommended for Jupyter)

```python
code = llm_feat.generate_features(df, metadata_df, mode='code')
print(code)
```

**Example Output:**
```python
import numpy as np

df['income_to_expense_ratio'] = np.where(df['expenses'] != 0, df['income'] / df['expenses'], np.nan)
df['savings_to_income_ratio'] = np.where(df['income'] != 0, df['savings'] / df['income'], np.nan)
df['net_savings'] = df['savings'] - df['expenses']
df['age_squared'] = df['age'] ** 2
df['income_per_age'] = np.where(df['age'] != 0, df['income'] / df['age'], np.nan)
```

> **Note**: In Jupyter notebooks, the code is automatically injected into the next cell.

#### Mode 2: Direct Feature Addition

```python
df_with_features = llm_feat.generate_features(df, metadata_df, mode='direct')
print(df_with_features.head())
```

**Example Output:**
```
   age  income  savings  expenses  income_to_expense_ratio  savings_to_income_ratio  net_savings  age_squared  income_per_age
0   25   50000    10000     40000                 1.250000                 0.200000       -30000          625          2000.0
1   30   60000    15000     45000                 1.333333                 0.250000       -30000          900          2000.0
2   35   70000    20000     50000                 1.400000                 0.285714       -30000         1225          2000.0
```

## 📖 Usage Examples

See [test_llm_feat.ipynb](test_llm_feat.ipynb) for complete usage examples in Jupyter notebook format.

> **Note**: The notebook examples use `df` as the DataFrame variable name, which is required for the generated code to work directly.


## Development Installation
### Prerequisites
- Python 3.10.19 or higher (tested with Python 3.10.19)
- [Poetry](https://python-poetry.org/docs/#installation) (for development)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (optional, for environment management)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/codeastra2/llm-feat.git
   cd llm-feat
   ```
2. **Create conda environment**:
   ```bash
   conda create -n llm_feat_310 python=3.10.19 -y;
   conda activate llm_feat_310
   ```
3. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
4. **Install dependencies**:
   ```bash
   poetry install
   ```
5. **Add env to jupyter kernels(to be used in ipython notebooks)**:
   ```bash
   poetry run python -m ipykernel install --user --name llm_feat_310 --display-name "Python (llm_feat_310)"
   ```
###  Running Tests
   ```bash
   poetry run pytest
   ```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please check our GitHub repository for guidelines.

## 👤 Author

**Srinivas Kumar** - [srinivas1996kumar@gmail.com](mailto:srinivas1996kumar@gmail.com)

## 📚 Documentation
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
For any questions or issues, please open an issue on our GitHub repository.


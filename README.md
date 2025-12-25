# llm-feat

[![Python Version](https://img.shields.io/badge/python-3.10.19%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/codeastra2/llm-feat)

llm-feat is a Python library that brings the power of Large Language Models (LLMs) to automated feature engineering for tabular data. The library leverages LLMs' capabilities to automatically generate and implement meaningful features for your machine learning tasks using GPT-4.

## 📑 Table of Contents

- [🌟 Key Features](#-key-features)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Examples](#-usage-examples)
- [📋 Requirements & Limitations](#-requirements--limitations-mvp)
- [✅ Running Tests](#-running-tests)
- [📄 License](#-license)
- [🤝 Contributing](#-contributing)
- [👤 Author](#-author)
- [📚 Documentation](#-documentation)

## 🌟 Key Features

- 🤖 **LLM-Powered**: Uses GPT-4 to generate intelligent feature engineering code
- 📊 **Pandas Integration**: Works seamlessly with pandas DataFrames
- 📝 **Dual Modes**: Get code suggestions or directly add features
- 🔧 **Jupyter Support**: Automatically injects code into next cell in Jupyter notebooks
- 🎯 **Metadata-Driven**: Uses column descriptions to generate contextually relevant features
- 🔀 **Categorical Support**: Automatic encoding strategies for categorical features
- 🎯 **Target-Aware**: Generates features specifically relevant to your prediction task

## 📦 Installation

### Prerequisites

- Python 3.10.19 or higher (tested with Python 3.10.19)
- [Poetry](https://python-poetry.org/docs/#installation) (for development)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (optional, for environment management)

### Production Installation

To install the latest release of llm-feat from PyPI:

```bash
pip install llm-feat
```

This will install the library and its core dependencies:
- pandas 2.0.3
- numpy 1.24.4
- openai >=1.0.0
- ipython >=8.0.0

### Development Installation

#### Option 1: Using Poetry (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/codeastra2/llm-feat.git
   cd llm-feat
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the Poetry shell**:
   ```bash
   poetry shell
   ```

#### Option 2: Using Conda + Poetry

1. **Create and activate conda environment**:
   ```bash
   conda create -n llm_feat python=3.10.19
   conda activate llm_feat
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Configure Poetry to use conda environment**:
   ```bash
   poetry config virtualenvs.create false
   ```

4. **Install dependencies**:
   ```bash
   cd llm-feat
   poetry install
   ```

5. **Set up Jupyter kernel** (optional, for notebook support):
   ```bash
   poetry run python -m ipykernel install --user --name llm_feat --display-name "Python (llm_feat)"
   ```

#### Option 3: Using pip

```bash
git clone https://github.com/yourusername/llm-feat.git
cd llm-feat
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Using Poetry
poetry run python -c "import llm_feat; print(f'llm-feat version: {llm_feat.__version__}')"

# Or using pip/conda
python -c "import llm_feat; print(f'llm-feat version: {llm_feat.__version__}')"
```

## 🚀 Quick Start

### 1. Set your OpenAI API Key

**Recommended: Use environment variable** (most secure):
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Then in Python:
```python
import os
import llm_feat

llm_feat.set_api_key(os.getenv("OPENAI_API_KEY"))
```

**Alternative: Set directly** (for testing only):
```python
import llm_feat

llm_feat.set_api_key("your-openai-api-key-here")
```

> ⚠️ **Security Note**: Never commit API keys to version control. Always use environment variables or secure secret management in production.

### 2. Prepare your Data and Metadata

```python
import pandas as pd

# Your dataset
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'savings': [10000, 15000, 20000, 25000, 30000]
})

# Metadata DataFrame with column information
metadata_df = pd.DataFrame({
    'column_name': ['age', 'income', 'savings'],
    'description': [
        'Age of the person in years',
        'Annual income in dollars',
        'Total savings in dollars'
    ],
    'data_type': ['numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, None]  # None for non-label columns
})
```

### 3. Generate Features

#### Mode 1: Get Code Suggestions (for Jupyter or manual review)

```python
# In Jupyter: Code will be injected into next cell automatically
# In regular Python: Returns code as string
code = llm_feat.generate_features(df, metadata_df, mode='code')
print(code)
```

> ⚠️ **Requirements**: 
> - Your DataFrame **must be named `df`** for the generated code to work directly. If your DataFrame has a different name, you'll need to manually replace `df` with your variable name in the generated code.
> - **Jupyter notebooks** are recommended for the best experience (automatic code injection). Regular Python scripts will receive the code as a string.
> 
> *Note: These requirements will be addressed in upcoming releases.*

#### Mode 2: Directly Add Features

```python
# Automatically adds features to DataFrame
df_with_features = llm_feat.generate_features(df, metadata_df, mode='direct')
print(df_with_features.head())
```

> **Note**: In `direct` mode, the DataFrame variable name doesn't matter as execution is handled internally.

## 📖 Usage Examples

See [test_llm_feat.ipynb](test_llm_feat.ipynb) for complete usage examples in Jupyter notebook format, including:
- Numerical feature engineering
- Categorical feature engineering
- Working with target columns
- Both code generation and direct feature addition modes

> **Note**: The notebook examples use `df` as the DataFrame variable name, which is required for the generated code to work directly.

## 📋 Requirements & Limitations (MVP)

### Requirements

⚠️ **Important Requirements**:
- **DataFrame Variable Name**: Generated code uses `df` as the DataFrame variable name. For `code` mode, your DataFrame must be named `df`, or you'll need to manually replace `df` with your variable name in the generated code. This will be addressed in upcoming releases.
- **Jupyter Notebooks**: The automatic code injection feature works best in Jupyter notebooks. While the package can be used in regular Python scripts, the seamless code injection experience is optimized for Jupyter environments. Support for other environments will be improved in future releases.

### Limitations

- No feature validation/quality checks yet
- Uses fixed prompt templates
- Categorical encoding strategies are automatically chosen by LLM (no manual control)

## ✅ Running Tests

To run the test suite, ensure you have the development dependencies installed and execute:

```bash
# Using Poetry
poetry run pytest

# Or using pytest directly
pytest tests/
```

Tests are located in the `tests/` directory and cover the core functionality of llm-feat.

You can also run individual test files:

```bash
python tests/test_imports.py
python tests/test_basic.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- **Report Bugs**: If you find a bug, please open an issue with a detailed description.
- **Suggest Features**: Have an idea for a new feature? Open an issue to discuss it.
- **Submit Pull Requests**: We love PRs! Here's how to submit one:
  1. Fork the repository
  2. Create a new branch for your feature
  3. Make your changes
  4. Submit a pull request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/codeastra2/llm-feat.git
   cd llm-feat
   ```

2. **Set up environment** (choose one method from Installation section above)

3. **Install development dependencies**:
   ```bash
   # Using Poetry
   poetry install
   
   # Or using pip
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   # Using Poetry
   poetry run pytest tests/ -v
   
   # Or using pytest directly
   pytest tests/ -v
   
   # Run basic structure tests
   poetry run python tests/test_basic.py
   ```

5. **Format code**:
   ```bash
   poetry run black .
   poetry run isort .
   ```

6. **Set up Jupyter notebook** (for testing):
   ```bash
   # If using Poetry with conda
   poetry run python -m ipykernel install --user --name llm_feat --display-name "Python (llm_feat)"
   
   # Then open test_llm_feat.ipynb and select the "Python (llm_feat)" kernel
   ```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Please ensure your code passes all checks before submitting a PR.

## 👤 Author

**Srinivas Kumar** - [srinivas1996kumar@gmail.com](mailto:srinivas1996kumar@gmail.com)

## 📚 Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
- [SECURITY.md](SECURITY.md) - Security policy and considerations
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

For any questions or issues, please open an issue on our GitHub repository.

## Acknowledgments

- Built with OpenAI's GPT-4
- Inspired by the need for automated feature engineering in ML workflows

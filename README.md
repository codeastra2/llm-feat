# llm-feat

[![Python Version](https://img.shields.io/badge/python-3.10.19%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

llm-feat is a Python library that automatically generates feature engineering code for pandas datasets using Large Language Models. Generate context-aware, target-specific features that can be automatically added to your DataFrame or customized before execution. 

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
Consider the problem of healthy and unhealthy labelling:
```python
# Your dataset with target column
df = pd.DataFrame({
    'height': [170, 175, 180, 165, 185, 172, 178, 168, 182, 174],
    'weight': [70, 75, 80, 65, 85, 72, 78, 68, 83, 74],
    'bmi': [24.2, 24.5, 24.7, 23.9, 24.8, 24.3, 24.6, 24.1, 25.0, 24.4],
    'health_score': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # Target: 1=healthy, 0=unhealthy
})

# Metadata with column descriptions and target definition
metadata_df = pd.DataFrame({
    'column_name': ['height', 'weight', 'bmi', 'health_score'],
    'description': [
        'Height in centimeters',
        'Weight in kilograms',
        'Body Mass Index',
        'Health classification score'
    ],
    'data_type': ['numeric', 'numeric', 'numeric', 'numeric'],
    'label_definition': [None, None, None, '1 if healthy, 0 if unhealthy']
})
```

### 3. Generate Features

#### Mode 1: Get Code (Recommended for Jupyter)
We will now generate features for the above dataset based on feature descriptions in the metadata and prior knowledge of the LLM.

```python
code = llm_feat.generate_features(df, metadata_df, mode='code', model='gpt-4o-mini')
print(code)
```

**Example Output:**
```python
import numpy as np

df['height_weight_ratio'] = df['height'] / df['weight'].replace(0, np.nan)
df['bmi_squared'] = df['bmi'] ** 2
df['weight_bmi_interaction'] = df['weight'] * df['bmi']
df['health_score_bmi_diff'] = df['health_score'] - df['bmi'].apply(lambda x: 1 if x < 24.9 else 0)
df['height_bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, np.inf], 
                                    labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
```

> **Note**: In Jupyter notebooks, the code is automatically injected into the next cell. The LLM generates target-aware features that are relevant to predicting `health_score`.

#### Mode 2: Direct Feature Addition

```python
df_with_features = llm_feat.generate_features(df, metadata_df, mode='direct', model='gpt-4o-mini')
print(df_with_features.head())
```

**Example Output:**
```
   height  weight   bmi  health_score  height_weight_ratio  bmi_squared  \
0     170      70  24.2             1             2.428571      585.64   
1     175      75  24.5             1             2.333333      600.25   
2     180      80  24.7             0             2.250000      610.09   

   weight_bmi_interaction  health_score_bmi_diff height_bmi_category  
0                1694.0                      0              Normal  
1                1837.5                      0              Normal  
2                1976.0                      0              Normal  
```

## 📖 Usage Examples

### Jupyter Notebook Example

See [example_llm_feat.ipynb](example_llm_feat.ipynb) for complete usage examples in Jupyter notebook format.

### Python Script Example

You can also run the example as a standalone Python script:

```bash
poetry run python example_llm_feat.py
```
> **Note**: The notebook and script examples use `df` as the DataFrame variable name, which is required for the generated code to work directly.


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


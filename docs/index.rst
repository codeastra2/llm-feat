llm-feat Documentation
======================

Automatically generate feature engineering code for pandas DataFrames using LLMs. Get context-aware, target-specific features that understand your domain.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   changelog

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install llm-feat

Quick Start
-----------

.. code-block:: python

   import pandas as pd
   import llm_feat

   llm_feat.set_api_key("your-openai-api-key")

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

Key Features
------------

* **Context-aware**: Uses column descriptions to generate relevant features
* **Target-aware**: Generates features specific to your prediction task
* **Categorical support**: Automatic encoding for categorical columns
* **Jupyter integration**: Code auto-injected into next cell
* **Feature reports**: Understand the reasoning behind each feature

Links
-----

* `GitHub Repository <https://github.com/codeastra2/llm-feat>`_
* `PyPI Package <https://pypi.org/project/llm-feat/>`_


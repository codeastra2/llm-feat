Quick Start
============

Setup
-----

First, set your OpenAI API key:

.. code-block:: python

   import pandas as pd
   import llm_feat

   llm_feat.set_api_key("your-openai-api-key")
   # Or use environment variable: export OPENAI_API_KEY="your-key"

Prepare Your Data
-----------------

Create your DataFrame and metadata:

.. code-block:: python

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

Generate Features
----------------

Get code suggestions (recommended for Jupyter):

.. code-block:: python

   code = llm_feat.generate_features(df, metadata_df, mode='code')
   print(code)

Or add features directly:

.. code-block:: python

   df_with_features = llm_feat.generate_features(
       df, metadata_df, mode='direct', model='gpt-4o-mini'
   )

Feature Reports
---------------

Get detailed explanations of generated features:

.. code-block:: python

   code, report = llm_feat.generate_features(
       df, metadata_df, mode='code', return_report=True
   )
   print(report)

The report includes domain understanding and explanations for each feature.


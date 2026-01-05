Examples
=========

Jupyter Notebook Examples
--------------------------

See the `example_llm_feat.ipynb <https://github.com/codeastra2/llm-feat/blob/main/example_llm_feat.ipynb>`_ notebook for complete usage examples.

Basic Example
-------------

.. code-block:: python

   import pandas as pd
   import llm_feat

   llm_feat.set_api_key("your-openai-api-key")

   # Create sample data
   df = pd.DataFrame({
       'height': [170, 175, 180, 165, 185],
       'weight': [70, 75, 80, 65, 85],
       'bmi': [24.2, 24.5, 24.7, 23.9, 24.8],
       'health_score': [1, 1, 0, 1, 0]
   })

   # Create metadata
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

   # Generate features
   code = llm_feat.generate_features(df, metadata_df, mode='code')
   print(code)

Feature Report Example
----------------------

.. code-block:: python

   code, report = llm_feat.generate_features(
       df,
       metadata_df,
       mode='code',
       return_report=True
   )

   print("Generated Code:")
   print(code)
   print("\nFeature Report:")
   print(report)

Categorical Features Example
-----------------------------

.. code-block:: python

   df = pd.DataFrame({
       'category': ['A', 'B', 'A', 'C', 'B'],
       'sales': [100, 200, 150, 300, 250],
       'target': [1, 0, 1, 0, 1]
   })

   metadata_df = pd.DataFrame({
       'column_name': ['category', 'sales', 'target'],
       'description': ['Product category', 'Sales amount', 'Target variable'],
       'data_type': ['categorical', 'numeric', 'numeric'],
       'label_definition': [None, None, 'Binary target']
   })

   code = llm_feat.generate_features(df, metadata_df, mode='code')
   print(code)


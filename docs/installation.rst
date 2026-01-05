Installation
============

Install from PyPI
------------------

The easiest way to install llm-feat is using pip:

.. code-block:: bash

   pip install llm-feat

Development Installation
------------------------

For development, clone the repository and install with Poetry:

.. code-block:: bash

   git clone https://github.com/codeastra2/llm-feat.git
   cd llm-feat
   conda create -n llm_feat_310 python=3.10.19 -y
   conda activate llm_feat_310
   poetry install

Requirements
------------

* Python 3.10.19 or higher
* pandas 2.0.3
* numpy 1.24.4
* openai >= 1.0.0
* ipython >= 8.0.0


# Documentation

This directory contains the Sphinx documentation for llm-feat.

## Building Locally

To build the documentation locally:

```bash
# Install documentation dependencies
poetry install --with docs

# Build the documentation
cd docs
poetry run sphinx-build -b html . _build/html

# View the documentation
open _build/html/index.html
```

## Read the Docs

The documentation is automatically built and hosted on Read the Docs at:
https://llm-feat.readthedocs.io/


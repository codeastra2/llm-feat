# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2025-01-XX

### Changed
- Updated Python version requirement to `>=3.10.0` for better compatibility
- Improved API documentation reference in README

## [0.2.2] - 2025-01-XX

### Added
- Sphinx documentation for Read the Docs
- API documentation with complete parameter reference
- Feature report generation documentation
- Updated README with concise quick start guide

### Changed
- Updated author information to use GitHub profile link
- Improved documentation structure and examples

## [0.2.1] - 2025-01-04

### Added
- Categorical feature engineering support
  - Automatic detection of categorical columns from metadata
  - Extraction of unique values and value counts for categorical columns
  - LLM-generated encoding strategies: one-hot encoding, target encoding, frequency encoding, label encoding
  - Group statistics features (numerical aggregations by category)
  - Interaction features between categorical and numerical columns
  - Support for multiple categorical data_type values: 'categorical', 'category', 'cat', 'string', 'text', 'object'

## [0.1.0] - 2024-XX-XX

### Added
- Initial MVP release
- Core `generate_features()` function with dual modes ('code' and 'direct')
- `set_api_key()` function for API key management
- Support for OpenAI GPT-4/GPT-4o models
- Metadata-driven feature generation
- Jupyter notebook integration with automatic code injection
- DataFrame analysis and context preparation for LLM
- Metadata validation
- Example usage scripts
- Basic test suite
- Comprehensive documentation (README, QUICKSTART, MVP_SUMMARY)

### Features
- Generate feature engineering code using LLMs
- Direct feature addition to pandas DataFrames
- Support for numerical feature engineering
- Target column awareness for supervised learning tasks
- Safe code execution with restricted globals

### Limitations (MVP Scope)
- Numerical features only
- No feature validation/quality checks
- Fixed prompt templates
- Basic code execution safety

[0.1.0]: https://github.com/codeastra2/llm-feat/releases/tag/v0.1.0


# Contributing to llm-feat

Thank you for your interest in contributing to llm-feat! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (Python version, OS, etc.)
- Any relevant error messages or logs

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- A clear description of the feature
- Use cases and examples
- Why this feature would be useful

### Code Contributions

1. **Fork the repository** and clone it locally
2. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Set up development environment**:
   ```bash
   conda activate llm_feat
   pip install -e ".[dev]"  # If dev dependencies are added
   ```
4. **Make your changes**:
   - Follow the existing code style
   - Add docstrings for new functions/classes
   - Add tests for new functionality
5. **Test your changes**:
   ```bash
   python test_imports.py
   python test_basic.py
   # Run any other relevant tests
   ```
6. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```
7. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Keep functions focused and small
- Write clear, descriptive variable names

## Testing

- Add tests for new features
- Ensure existing tests still pass
- Test edge cases and error conditions

## Documentation

- Update README.md if adding new features
- Add examples for new functionality
- Update docstrings for any API changes

## Questions?

Feel free to open an issue with questions or reach out to the maintainers.

Thank you for contributing! 🎉


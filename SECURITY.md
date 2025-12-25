# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Considerations

### API Key Management

⚠️ **CRITICAL**: Never commit API keys to version control.

- Always use environment variables: `export OPENAI_API_KEY="your-key"`
- Never hardcode API keys in notebooks, scripts, or configuration files
- If you accidentally commit an API key:
  1. Rotate/revoke the exposed key immediately
  2. Remove it from git history using `git filter-branch` or BFG Repo-Cleaner
  3. Force push to remote (if already pushed)

### Code Execution

This package executes LLM-generated Python code using Python's `exec()` function. While we implement safety measures, users should be aware of the following:

1. **Code Review Recommended**: Always review generated code before executing, especially in production environments
2. **Restricted Execution**: We use restricted globals to limit available functions, but this is not a complete sandbox
3. **Use Code Mode**: Consider using `mode='code'` to review code before execution instead of `mode='direct'`

### Best Practices

- ✅ Review generated code before executing
- ✅ Test generated features on sample data first
- ✅ Use in trusted environments only
- ✅ Never execute code from untrusted sources
- ❌ Don't use with sensitive data without review
- ❌ Don't execute in production without validation

### Reporting Security Issues

If you discover a security vulnerability, please email [your-email@example.com] instead of opening a public issue.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue.

## Future Security Enhancements

Planned improvements:
- AST validation before execution
- More robust sandboxing
- Code sanitization
- Execution timeouts
- Resource limits


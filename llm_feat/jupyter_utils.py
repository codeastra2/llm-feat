"""Utilities for Jupyter notebook integration"""


def is_jupyter() -> bool:
    """Check if running in Jupyter/IPython environment"""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        return ipython is not None and ipython.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def inject_code_to_next_cell(code: str) -> bool:
    """
    Inject code into the next Jupyter cell.
    Works with both JupyterLab and classic Jupyter Notebook.

    Args:
        code: Python code string to inject

    Returns:
        True if successful, False otherwise
    """
    if not is_jupyter():
        return False

    try:
        import json

        from IPython import get_ipython
        from IPython.display import Javascript, display

        ipython = get_ipython()
        injected = False

        # Method 1: Try JavaScript for immediate cell creation
        # (works in classic Notebook)
        try:
            escaped_code = json.dumps(code)

            js_code = f"""
            (function() {{
                var code = {escaped_code};

                // Try classic Jupyter Notebook API
                if (typeof IPython !== 'undefined' && IPython.notebook) {{
                    var cell = IPython.notebook.insert_cell_below('code');
                    cell.set_text(code);
                    cell.focus_cell();
                    return true;
                }}

                // Try Jupyter API
                if (typeof Jupyter !== 'undefined' && Jupyter.notebook) {{
                    var cell = Jupyter.notebook.insert_cell_below('code');
                    cell.set_text(code);
                    cell.focus_cell();
                    return true;
                }}

                return false;
            }})();
            """

            display(Javascript(js_code))
            injected = True
            print("✓ Attempted to create new cell with code - check below")

        except Exception:
            pass

        # Method 2: Use IPython's set_next_input (works in all
        # environments)
        # This sets the code for the next cell that gets created
        try:
            ipython.set_next_input(code, replace=False)
            if not injected:
                print(
                    "✓ Code ready for next cell - create a new cell below "
                    "and it will appear automatically"
                )
            else:
                print("  (Also set as next input - will appear in next cell " "you create)")
            return True
        except Exception:
            if injected:
                return True  # JavaScript worked, even if set_next_input
                # failed
            pass

    except Exception as e:
        print(f"⚠️  Could not automatically inject code: {e}")
        print("The generated code is displayed above - you can copy it " "manually.")
        return False

    return False


def get_code_string(code: str) -> str:
    """
    Return code as a formatted string for display/copying.

    Args:
        code: Python code string

    Returns:
        Formatted code string
    """
    return f"\n# Generated Feature Engineering Code\n{code}\n"

"""
llm-feat: Automated feature engineering using LLMs
"""

from .core import set_api_key, generate_features
from .version import __version__

__all__ = ["set_api_key", "generate_features", "__version__"]


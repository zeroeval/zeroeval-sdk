"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""
from .core import exp

# Explicitly declare public API
__all__ = ["exp"]

# Optional: Add version info
__version__ = "0.1.0"
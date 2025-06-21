"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""
from .core import Dataset, Experiment, experiment, init
from .observability import span

# Explicitly declare public API
__all__ = ["Dataset", "Experiment", "experiment", "init", "span"]

# Optional: Add version info
__version__ = "0.1.0"
"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""
from .core import Dataset, Experiment, experiment

# Explicitly declare public API
__all__ = ["Dataset", "Experiment", "experiment"]

# Optional: Add version info
__version__ = "0.1.0"
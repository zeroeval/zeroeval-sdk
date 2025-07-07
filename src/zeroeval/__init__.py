"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""

from .core import Dataset, Experiment, experiment, init
from .observability import span
from .observability.signals import set_signal
from .observability.span import Span
from .observability.tracer import tracer

# Import helper methods directly from tracer for convenience
get_current_span = tracer.get_current_span
get_current_trace = tracer.get_current_trace
get_current_session = tracer.get_current_session

# Method for setting tags
set_tag = tracer.set_tag

# Explicitly declare public API
__all__ = [
    "Dataset",
    "Experiment",
    "experiment",
    "get_current_session",
    "get_current_span",
    "get_current_trace",
    "init",
    "set_signal",
    "set_tag",
    "span",
]

# Optional: Add version info
__version__ = "0.1.0"

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
    "init", 
    "span",
    "get_current_span", 
    "get_current_trace", 
    "get_current_session", 
    "set_tag",
    "set_signal"
]

# Optional: Add version info
__version__ = "0.1.0"
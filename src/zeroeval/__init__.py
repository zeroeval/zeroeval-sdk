"""
ZeroEval SDK - AI experiment tracking and observability.

Main API:
    ze.init() - Initialize the SDK
    @ze.span() - Decorator for tracing functions
    ze.span() - Context manager for manual spans
    ze.start_span() - Alternative way to start spans
    ze.get_current_span() - Get the current active span
    ze.get_current_trace() - Get the current trace ID
    ze.get_current_session() - Get the current session ID
    ze.set_tag() - Attach tags to a span, trace, or session
"""

# Core functionality imports
from .core.dataset_class import Dataset
from .core.task import task
from .core.evaluation import evaluation
from .core.metrics import column_metric, run_metric
from .core.init import init

# Provider imports
from .providers import ZeroEvalOTLPProvider, SingleProcessorProvider, langfuse_zeroeval

# Observability imports - import the actual classes/objects
from .observability import tracer, span as _SpanClass

# Create convenience functions that match the expected API
def start_span(name: str, **kwargs):
    """Start a new span. Convenience wrapper around tracer.start_span()."""
    return tracer.start_span(name=name, **kwargs)

def get_current_span():
    """Get the current active span."""
    return tracer.get_current_span()

def get_current_trace():
    """Get the current trace ID."""
    return tracer.get_current_trace()

def get_current_session():
    """Get the current session ID."""
    return tracer.get_current_session()

def set_tag(target, tags):
    """Attach tags to a Span, trace or session.
    
    Args:
        target: Can be a Span instance, trace_id string, or session_id string
        tags: Dictionary of tags to apply
    """
    return tracer.set_tag(target, tags)

# Use the imported span class directly
span = _SpanClass

# Define what's exported
__all__ = [
    # Core functionality
    "Dataset", 
    "task", 
    "evaluation", 
    "column_metric", 
    "run_metric", 
    "init",
    # Providers
    "ZeroEvalOTLPProvider", 
    "SingleProcessorProvider", 
    "langfuse_zeroeval",
    # Observability
    "tracer",
    "span",
    "start_span",
    "get_current_span",
    "get_current_trace",
    "get_current_session",
    "set_tag",
]

# Version info
__version__ = "0.6.9"
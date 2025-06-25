"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""
from .core import Dataset, Experiment, experiment, init
from .observability import span
from .observability.tracer import tracer
from .observability.span import Span

# Explicitly declare public API
__all__ = ["Dataset", "Experiment", "experiment", "init", "span"]

# Optional: Add version info
__version__ = "0.1.0"

# ------------------------------------------------------------------
# Tag helpers re-exported at package root for convenience
# ------------------------------------------------------------------

def get_current_span():
    """Return the current active Span (or None)."""
    return tracer.current_span()


def get_current_trace():
    """Return the current trace_id (or None if no active span)."""
    current = tracer.current_span()
    return current.trace_id if current else None


def get_current_session():
    """Return the current session_id (or None if no active span)."""
    current = tracer.current_span()
    return current.session_id if current else None


def set_tag(target, tags: dict):
    """Attach *tags* to a Span, trace or session.

    * ``target`` can be:
        - a ``Span`` instance → tags are applied to that span (and will bubble to children)
        - a ``str`` trace_id      → tags applied to every span in that trace
        - a ``str`` session_id    → tags applied to every span within that session
    """
    if not isinstance(tags, dict):
        raise TypeError("tags must be a dictionary")

    if isinstance(target, Span):
        target.tags.update(tags)
    elif isinstance(target, str):
        # Heuristically decide whether this is a trace_id or session_id
        if tracer.is_active_trace(target):
            tracer.add_trace_tags(target, tags)
        else:
            tracer.add_session_tags(target, tags)
    else:
        raise TypeError("Unsupported target type for set_tag")

# Extend public API
__all__.extend(["get_current_span", "get_current_trace", "get_current_session", "set_tag"])
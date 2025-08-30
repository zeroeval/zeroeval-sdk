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
from .providers import ZeroEvalOTLPProvider, SingleProcessorProvider

# Observability imports - import the actual classes/objects
from .observability import tracer, span as _SpanClass, zeroeval_prompt
from .types import Prompt
from .client import ZeroEval as PromptClient

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

# Create convenience alias for zeroeval_prompt
prompt = zeroeval_prompt

# Prompt library convenience wrappers
from typing import Optional

_prompt_client_instance: Optional[PromptClient] = None


def _ensure_prompt_client() -> PromptClient:
    global _prompt_client_instance
    if _prompt_client_instance is None:
        _prompt_client_instance = PromptClient()
    return _prompt_client_instance


def get_prompt(
    slug: str,
    *,
    version=None,
    tag=None,
    fallback=None,
    variables=None,
    task_name: None | str = None,
    render: bool = True,
    missing: str = "error",
    use_cache: bool = True,
    timeout: Optional[float] = None,
):
    client = _ensure_prompt_client()
    return client.get_prompt(
        slug,
        version=version,
        tag=tag,
        fallback=fallback,
        variables=variables,
        task_name=task_name,
        render=render,
        missing=missing,
        use_cache=use_cache,
        timeout=timeout,
    )


class _PromptsNamespace:
    def get(self, slug: str, **kwargs):
        return get_prompt(slug, **kwargs)


prompts = _PromptsNamespace()

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

    # Observability
    "tracer",
    "span",
    "start_span",
    "get_current_span",
    "get_current_trace",
    "get_current_session",
    "set_tag",
    # Prompt utilities
    "zeroeval_prompt",
    "prompt",
    # Prompt library
    "Prompt",
    "PromptClient",
    "get_prompt",
    "prompts",
]

# Version info
__version__ = "0.6.9"
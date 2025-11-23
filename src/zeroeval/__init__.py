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
    ze.choose() - Make weighted A/B test choices
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
from .observability.choice import choose
from .observability.signals import set_signal
from .types import Prompt
from .client import ZeroEval as PromptClient
from .utils.hash import sha256_hex
import re
from .errors import PromptNotFoundError, PromptRequestError

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

# Version-aware prompt wrapper
def prompt(
    name: str,
    *,
    content: "Optional[str]" = None,
    variables: "Optional[dict]" = None,
    from_: "Optional[str]" = None,
    **kwargs,
) -> str:
    """
    Version-aware prompt helper integrated with Prompt Library.

    Exactly one of `content` or `from` must be provided. If `from` is "latest",
    fetch the latest version for the task-attached prompt. Otherwise `from` must be
    a 64-char lowercase hex SHA-256 content hash.

    For backward compatibility, `from_` is still accepted and behaves the same as `from`.
    """
    # Accept `from` via kwargs and map to `from_`
    if kwargs:
        if "from" in kwargs:
            if from_ is not None:
                raise ValueError("Provide only one of 'from' or 'from_'")
            from_ = kwargs.pop("from")
        if kwargs:
            raise TypeError("Unexpected keyword arguments: " + ", ".join(kwargs.keys()))
    if (content is None and from_ is None) or (content is not None and from_ is not None):
        raise ValueError("Provide exactly one of 'content' or 'from'")

    client = _ensure_prompt_client()

    # Ensure/fetch the prompt version via backend
    if content is not None:
        content_hash = sha256_hex(content)
        prompt_obj = client.ensure_task_prompt_version(task_name=name, content=content, content_hash=content_hash)
    else:
        assert from_ is not None
        if from_ == "latest":
            try:
                prompt_obj = client.get_task_prompt_latest(task_name=name)
            except PromptNotFoundError as _e:
                raise PromptRequestError(
                    f"No prompt versions found for task '{name}'. "
                    f"Create one with ze.prompt(name, content=...) or publish a version in the Prompt Library."
                )
        else:
            if not re.fullmatch(r"[0-9a-f]{64}", from_):
                raise ValueError("from must be 'latest' or a 64-char lowercase hex SHA-256 hash")
            prompt_obj = client.get_task_prompt_version_by_hash(task_name=name, content_hash=from_)

    # Pull linkage metadata for decoration
    prompt_slug = None
    try:
        # Prefer metadata.prompt_slug if provided by server
        prompt_slug = (prompt_obj.metadata or {}).get("prompt_slug")
        if not prompt_slug:
            prompt_slug = (prompt_obj.metadata or {}).get("prompt")
    except Exception:
        prompt_slug = None

    return zeroeval_prompt(
        name=name,
        content=prompt_obj.content,
        variables=variables,
        prompt_slug=prompt_slug,
        prompt_version=prompt_obj.version,
        prompt_version_id=getattr(prompt_obj, "version_id", None),
        content_hash=content_hash if content is not None else None,
    )

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


def log_completion(
    *,
    prompt_slug: str,
    prompt_id: str,
    prompt_version_id: str,
    messages: list,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
    model_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    duration_ms: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    has_error: bool = False,
    error_message: Optional[str] = None,
    span_id: Optional[str] = None,
):
    """
    Log a completion for a specific prompt.
    
    This automatically tracks prompt usage without requiring manual wrapping.
    """
    client = _ensure_prompt_client()
    return client.log_completion(
        prompt_slug=prompt_slug,
        prompt_id=prompt_id,
        prompt_version_id=prompt_version_id,
        messages=messages,
        input_text=input_text,
        output_text=output_text,
        model_id=model_id,
        metadata=metadata,
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        has_error=has_error,
        error_message=error_message,
        span_id=span_id,
    )


def send_feedback(
    *,
    prompt_slug: str,
    completion_id: str,
    thumbs_up: bool,
    reason: Optional[str] = None,
    expected_output: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Send feedback for a specific completion.
    """
    client = _ensure_prompt_client()
    return client.send_feedback(
        prompt_slug=prompt_slug,
        completion_id=completion_id,
        thumbs_up=thumbs_up,
        reason=reason,
        expected_output=expected_output,
        metadata=metadata,
    )


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
    "set_signal",
    "choose",
    # Prompt utilities
    "zeroeval_prompt",
    "prompt",
    # Prompt library
    "Prompt",
    "PromptClient",
    "get_prompt",
    "prompts",
    # Completion logging and feedback
    "log_completion",
    "send_feedback",
]

# Version info
__version__ = "0.6.119"

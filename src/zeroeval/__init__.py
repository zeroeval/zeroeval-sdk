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
import logging as _logging
_prompt_logger = _logging.getLogger("zeroeval.prompt")

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

    When `content` is provided alone, it serves as a fallback - the SDK will automatically
    fetch the latest optimized version from the backend if one exists. This allows you
    to hardcode a default prompt while seamlessly using tuned versions in production.

    If `from_` is specified, it controls version behavior:
    - `from_="latest"` explicitly fetches the latest version (fails if none exists)
    - `from_="explicit"` always uses the provided `content` (bypasses auto-optimization, requires `content`)
    - `from_="<hash>"` fetches a specific version by its 64-char SHA-256 content hash

    For backward compatibility, `from_` is still accepted and behaves the same as `from`.
    """
    _prompt_logger.debug(f"=== ze.prompt() called ===")
    _prompt_logger.debug(f"  name: {name!r}")
    _prompt_logger.debug(f"  content provided: {content is not None} (length: {len(content) if content else 0})")
    _prompt_logger.debug(f"  variables: {list(variables.keys()) if variables else None}")
    _prompt_logger.debug(f"  from_: {from_!r}")
    
    # Accept `from` via kwargs and map to `from_`
    if kwargs:
        if "from" in kwargs:
            if from_ is not None:
                _prompt_logger.error("Both 'from' and 'from_' provided - this is an error")
                raise ValueError("Provide only one of 'from' or 'from_'")
            from_ = kwargs.pop("from")
            _prompt_logger.debug(f"  from_ (via kwargs): {from_!r}")
        if kwargs:
            _prompt_logger.error(f"Unexpected keyword arguments: {list(kwargs.keys())}")
            raise TypeError("Unexpected keyword arguments: " + ", ".join(kwargs.keys()))
    
    if content is None and from_ is None:
        _prompt_logger.error("Neither 'content' nor 'from' provided")
        raise ValueError("Must provide either 'content' or 'from'")
    
    # Validate that explicit requires content
    if from_ == "explicit" and content is None:
        _prompt_logger.error("from='explicit' requires 'content' to be provided")
        raise ValueError("from='explicit' requires 'content' to be provided")

    client = _ensure_prompt_client()
    content_hash = None
    prompt_obj = None

    # Priority order:
    # 1. If from_="explicit", always use the provided content (bypass auto-optimization)
    # 2. If from_ is specified (latest or hash), use it (strict mode)
    # 3. If only content is provided, try to fetch latest first, fall back to ensuring content
    try:
        if from_ == "explicit":
            # Explicit mode: always use the provided content, no auto-optimization
            _prompt_logger.info(f"Mode: EXPLICIT - using provided content directly for task '{name}'")
            content_hash = sha256_hex(content)
            _prompt_logger.debug(f"  content_hash: {content_hash}")
            prompt_obj = client.ensure_task_prompt_version(
                task_name=name,
                content=content,
                content_hash=content_hash
            )
            _prompt_logger.debug(f"  prompt version ensured: version={getattr(prompt_obj, 'version', 'N/A')}")
        elif from_ is not None:
            # Explicit from_ takes priority - strict mode for latest or hash
            if from_ == "latest":
                _prompt_logger.info(f"Mode: LATEST - fetching latest version for task '{name}'")
                try:
                    prompt_obj = client.get_task_prompt_latest(task_name=name)
                    _prompt_logger.debug(f"  latest version fetched: version={getattr(prompt_obj, 'version', 'N/A')}")
                except PromptNotFoundError as _e:
                    _prompt_logger.error(f"No prompt versions found for task '{name}'")
                    raise PromptRequestError(
                        f"No prompt versions found for task '{name}'. "
                        f"Create one with ze.prompt(name, content=...) or publish a version in the Prompt Library."
                    )
            else:
                if not re.fullmatch(r"[0-9a-f]{64}", from_):
                    _prompt_logger.error(f"Invalid hash format: {from_}")
                    raise ValueError("from must be 'latest', 'explicit', or a 64-char lowercase hex SHA-256 hash")
                _prompt_logger.info(f"Mode: HASH - fetching specific version by hash for task '{name}'")
                _prompt_logger.debug(f"  hash: {from_}")
                prompt_obj = client.get_task_prompt_version_by_hash(task_name=name, content_hash=from_)
                _prompt_logger.debug(f"  version fetched by hash: version={getattr(prompt_obj, 'version', 'N/A')}")
        elif content is not None:
            # Auto-tune mode: try latest first, fall back to content
            _prompt_logger.info(f"Mode: AUTO-TUNE - trying latest, falling back to provided content for task '{name}'")
            content_hash = sha256_hex(content)
            _prompt_logger.debug(f"  content_hash: {content_hash}")
            try:
                prompt_obj = client.get_task_prompt_latest(task_name=name)
                _prompt_logger.info(f"  SUCCESS: Using TUNED version {getattr(prompt_obj, 'version', 'N/A')} for task '{name}'")
            except (PromptNotFoundError, PromptRequestError) as e:
                _prompt_logger.debug(f"  No latest version found ({type(e).__name__}), ensuring provided content")
                # No latest version exists, ensure the provided content as a version
                prompt_obj = client.ensure_task_prompt_version(
                    task_name=name, 
                    content=content, 
                    content_hash=content_hash
                )
                _prompt_logger.info(f"  Using FALLBACK content for task '{name}' (version={getattr(prompt_obj, 'version', 'N/A')})")
    except Exception as e:
        _prompt_logger.error(f"Error fetching/ensuring prompt for task '{name}': {type(e).__name__}: {e}")
        raise

    # Pull linkage metadata for decoration
    prompt_slug = None
    try:
        # Prefer metadata.prompt_slug if provided by server
        prompt_slug = (prompt_obj.metadata or {}).get("prompt_slug")
        if not prompt_slug:
            prompt_slug = (prompt_obj.metadata or {}).get("prompt")
        _prompt_logger.debug(f"  prompt_slug: {prompt_slug!r}")
    except Exception as e:
        _prompt_logger.warning(f"  Could not extract prompt_slug from metadata: {e}")
        prompt_slug = None

    # Log the final prompt metadata that will be embedded
    version_id = getattr(prompt_obj, "version_id", None)
    _prompt_logger.info(f"=== ze.prompt() result for task '{name}' ===")
    _prompt_logger.info(f"  prompt_version: {prompt_obj.version}")
    _prompt_logger.info(f"  prompt_version_id: {version_id}")
    _prompt_logger.info(f"  prompt_slug: {prompt_slug}")
    _prompt_logger.info(f"  content_hash: {content_hash}")
    _prompt_logger.info(f"  content length: {len(prompt_obj.content)} chars")
    _prompt_logger.debug(f"  content preview: {prompt_obj.content[:100]}..." if len(prompt_obj.content) > 100 else f"  content: {prompt_obj.content}")

    result = zeroeval_prompt(
        name=name,
        content=prompt_obj.content,
        variables=variables,
        prompt_slug=prompt_slug,
        prompt_version=prompt_obj.version,
        prompt_version_id=version_id,
        content_hash=content_hash,
    )
    
    _prompt_logger.debug(f"  zeroeval_prompt returned string of length {len(result)}")
    _prompt_logger.debug(f"  result preview: {result[:150]}..." if len(result) > 150 else f"  result: {result}")
    
    return result

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

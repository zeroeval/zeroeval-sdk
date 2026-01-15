import atexit
import builtins
import contextlib
import logging
import os
import random
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional, Union

try:
    from opentelemetry import trace as otel_trace
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False

from .span import Span
from .writer import SpanBackendWriter, SpanWriter

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    """Represents a collection of spans and metadata for a single trace."""
    trace_id: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    ref_count: int = 0
    is_sampled: bool = True  # Whether this trace is being sampled


@dataclass
class Signal:
    """Represents a signal that can be attached to entities."""
    name: str
    value: Union[str, bool, int, float]
    signal_type: str = "boolean"  # "boolean" or "numerical"

    def __post_init__(self):
        # Auto-detect signal type and normalize value
        if isinstance(self.value, bool):
            self.signal_type = "boolean"
            self.value = "true" if self.value else "false"
        elif isinstance(self.value, (int, float)):
            self.signal_type = "numerical"
            self.value = str(self.value)
        else:
            # For string values, try to detect boolean
            str_val = str(self.value).lower()
            if str_val in ("true", "false"):
                self.signal_type = "boolean"
                self.value = str_val
            else:
                # Default to boolean for string values
                self.signal_type = "boolean"
                self.value = str(self.value)


def _normalize_session_id(session_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Normalize a session_id to UUID format.

    If the session_id is already a valid UUID, it's returned as-is.
    If it's a custom string, it's deterministically converted to a UUIDv5.

    Args:
        session_id: The original session ID (can be UUID or custom string)

    Returns:
        A tuple of (normalized_session_id, original_session_id_as_name).
        The second value is only set if the session_id was not a UUID and can
        be used as the session_name if no explicit name was provided.
    """
    if not session_id:
        return None, None

    try:
        # Check if it's already a valid UUID
        uuid.UUID(str(session_id))
        return str(session_id), None
    except ValueError:
        # Not a UUID - deterministically derive a UUIDv5 from the string
        normalized = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(session_id)))
        return normalized, str(session_id)


class Tracer:
    """
    Singleton tracer that manages spans, buffering, and flushing to writers.
    Spans are flushed periodically or when the buffer is full, without waiting
    for traces to complete, enabling real-time streaming.
    """
    _instance = None
    _lock = threading.Lock()

    _active_spans_ctx: ContextVar[list[Span]] = ContextVar("active_spans", default=[])

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initialize the tracer's internal state and register for graceful shutdown."""
        self._spans: list[dict[str, Any]] = []
        self._active_spans: dict[str, list[Span]] = {}  # For legacy/compatibility
        self._traces: dict[str, Trace] = {}  # Replaces _trace_buckets and _trace_counts
        self._last_flush_time = time.time()
        self._writer: Optional[SpanWriter] = None  # Lazy-loaded to allow env vars to be set first
        self._flush_interval: float = 1.0  # Flush more frequently for streaming
        self._max_spans: int = 20
        self._flush_lock = threading.Lock()
        self._integrations: dict[str, Any] = {}

        # Sampling rate configuration (0.0 to 1.0, where 1.0 means sample everything)
        sampling_env = os.environ.get("ZEROEVAL_SAMPLING_RATE", "1.0")
        try:
            self._sampling_rate = float(sampling_env)
            self._sampling_rate = max(0.0, min(1.0, self._sampling_rate))  # Clamp to [0, 1]
        except ValueError:
            logger.warning(f"Invalid ZEROEVAL_SAMPLING_RATE value: {sampling_env}. Using default 1.0")
            self._sampling_rate = 1.0

        # Async signal writer (optional)
        self._async_signal_enabled = False
        self._signal_writer = None

        # Config for integrations, read from environment variable first
        self._integrations_config: dict[str, bool] = {}
        disabled_env = os.environ.get("ZEROEVAL_DISABLED_INTEGRATIONS", "")
        if disabled_env:
            disabled_names = {name.strip() for name in disabled_env.split(',') if name.strip()}
            for name in disabled_names:
                self._integrations_config[name] = False # Disable them
            logger.info(f"Integrations disabled via environment variable: {disabled_names}")

        self.collect_code_details: bool = True
        self._shutdown_called: bool = False
        self._shutdown_lock = threading.Lock()

        # Global tags applied to all spans/traces/sessions
        self._global_tags: dict[str, str] = {}

        # Containers for trace- and session-level tags
        self._trace_level_tags: dict[str, dict[str, str]] = {}
        self._session_level_tags: dict[str, dict[str, str]] = {}

        # Map trace ID to session info for OTEL context propagation
        self._trace_to_session: dict[str, dict[str, str]] = {}

        logger.info("Initializing tracer for streaming...")
        logger.info(f"Tracer config: flush_interval={self._flush_interval}s, max_spans={self._max_spans}, sampling_rate={self._sampling_rate}")

        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_periodically, daemon=True)
        self._flush_thread.start()

        # Don't auto-setup integrations here - wait for init() to be called
        self._integrations_initialized = False

        # Register shutdown hook
        atexit.register(self.shutdown)

    @property
    def writer(self) -> SpanWriter:
        """Lazy-load writer to ensure environment variables are available."""
        if self._writer is None:
            self._writer = SpanBackendWriter()
        return self._writer

    def ensure_integrations_initialized(self) -> None:
        """Ensure integrations are initialized, but only once."""
        if not self._integrations_initialized:
            # Re-read disabled integrations from environment in case they were set by init()
            disabled_env = os.environ.get("ZEROEVAL_DISABLED_INTEGRATIONS", "")
            if disabled_env:
                disabled_names = {name.strip() for name in disabled_env.split(',') if name.strip()}
                for name in disabled_names:
                    self._integrations_config[name] = False
                logger.info(f"Integrations disabled via environment variable: {disabled_names}")

            self._setup_available_integrations()
            self._integrations_initialized = True

    def _setup_available_integrations(self) -> None:
        """Automatically set up all available integrations."""
        # Import here to avoid circular imports
        from .integrations.gemini.integration import GeminiIntegration
        from .integrations.httpx.integration import HttpxIntegration
        from .integrations.langchain.integration import LangChainIntegration
        from .integrations.langgraph.integration import LangGraphIntegration
        from .integrations.openai.integration import OpenAIIntegration
        from .integrations.pydanticai.integration import PydanticAIIntegration
        from .integrations.vocode.integration import VocodeIntegration

        # List of all integration classes
        # Note: PydanticAIIntegration should be set up BEFORE OpenAIIntegration
        # so that the parent agent span is created before any OpenAI calls
        integration_classes = [
            PydanticAIIntegration, # Auto-instrument PydanticAI agents (must be before OpenAI)
            OpenAIIntegration,
            GeminiIntegration,     # Auto-instrument Gemini
            HttpxIntegration,      # Auto-instrument httpx for network-level tracing
            LangChainIntegration,  # Auto-instrument LangChain
            LangGraphIntegration,  # Auto-instrument LangGraph
            VocodeIntegration,     # Auto-instrument Vocode voice SDK
        ]

        logger.info(f"Checking for available integrations: {[i.__name__ for i in integration_classes]}")
        # Setup each available integration
        for integration_class in integration_classes:
            integration_name = integration_class.__name__
            # Check if integration is explicitly disabled in config. Default is enabled (None or True)
            if self._integrations_config.get(integration_name) is False:
                logger.info(f"Skipping disabled integration as per configuration: {integration_name}")
                continue

            if integration_class.is_available():
                try:
                    logger.info(f"Setting up integration: {integration_name}")
                    integration = integration_class(self)

                    # Use safe setup method with better error handling
                    if integration.safe_setup():
                        self._integrations[integration_name] = integration
                        logger.info(f"✅ Successfully set up integration: {integration_name}")
                    else:
                        setup_error = integration.get_setup_error()
                        logger.error(f"❌ Failed to set up integration {integration_name}: {setup_error}")
                        self._print_user_friendly_error(integration_name, setup_error)
                except Exception as exc:
                    logger.error(f"❌ Critical error setting up integration {integration_name}: {exc}", exc_info=True)
                    self._print_user_friendly_error(integration_name, exc)
            else:
                logger.info(f"Integration not available: {integration_name}")

        if self._integrations:
            logger.info(f"Active integrations: {list(self._integrations.keys())}")
        else:
            logger.info("No active integrations found.")

    def reinitialize_integrations(self):
        """Reinitialize integrations. Useful after init() sets up logging."""
        logger.info("Reinitializing integrations...")

        # Properly teardown existing integrations before clearing
        for integration_name, integration in self._integrations.items():
            try:
                logger.debug(f"Tearing down integration: {integration_name}")
                integration.teardown()
            except Exception as e:
                logger.error(f"Failed to teardown integration {integration_name}: {e}")

        self._integrations.clear()
        self._setup_available_integrations()

    def _print_user_friendly_error(self, integration_name: str, error: Exception) -> None:
        """Print user-friendly error messages for common integration issues."""
        error_str = str(error)

        if "builtin_function_or_method" in error_str and "__get__" in error_str:
            print(f"\n[ZeroEval] ❌ {integration_name} failed due to Python compatibility issue.")
            print("This often happens with Python 3.13+ and certain library versions.")
            print("💡 You can disable this integration with:")
            print("   import zeroeval as ze")
            print(f"   ze.tracer.configure(integrations={{'{integration_name}': False}})")
            print("   ze.init(api_key='your-key')")
        elif "typing.Generic" in error_str:
            print(f"\n[ZeroEval] ❌ {integration_name} failed due to type annotation compatibility issue.")
            print("This is typically caused by Python 3.13+ with older library versions.")
            print("💡 You can:")
            print(f"   1. Disable this integration: ze.tracer.configure(integrations={{'{integration_name}': False}})")
            print("   2. Or use Python 3.11 or 3.12 if possible")
        elif "ImportError" in error_str or "ModuleNotFoundError" in error_str:
            print(f"\n[ZeroEval] ❌ {integration_name} failed due to missing dependencies.")
            print("💡 Install required packages or disable this integration:")
            print(f"   ze.tracer.configure(integrations={{'{integration_name}': False}})")
        else:
            print(f"\n[ZeroEval] ❌ {integration_name} setup failed: {error}")
            print("💡 You can disable this integration:")
            print(f"   ze.tracer.configure(integrations={{'{integration_name}': False}})")
        print()  # Empty line for readability

    def is_shutting_down(self) -> bool:
        """Check if the tracer is currently in the process of shutting down."""
        with self._shutdown_lock:
            return self._shutdown_called

    def shutdown(self) -> None:
        """
        Gracefully shuts down the tracer. This method is registered via atexit
        to be called automatically on normal program termination, ensuring all
        buffered spans are sent.
        """
        with self._shutdown_lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        logger.info("Program exiting. Performing final flush of all remaining traces...")

        # Stop async signal writer if enabled
        if self._async_signal_enabled and self._signal_writer:
            try:
                import asyncio

                from .signal_writer import SignalWriterManager

                # Stop the signal writer
                def stop_writer():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(SignalWriterManager.stop_writer())

                stop_thread = threading.Thread(target=stop_writer)
                stop_thread.start()
                stop_thread.join(timeout=5.0)  # Wait up to 5 seconds

            except Exception as e:
                logger.error(f"Error stopping signal writer: {e}")

        # Teardown integrations safely
        for integration_name, integration in self._integrations.items():
            try:
                integration.teardown()
            except Exception:
                logger.error(f"Failed to teardown integration {integration_name}", exc_info=True)

        # Final flush for any remaining spans in the buffer
        self.flush()

    def __del__(self):
        """Cleanup integrations when tracer is destroyed."""
        for integration in self._integrations.values():
            with contextlib.suppress(builtins.BaseException):
                integration.teardown()

    def configure(self,
                  flush_interval: Optional[float] = None,
                  max_spans: Optional[int] = None,
                  collect_code_details: Optional[bool] = None,
                  integrations: Optional[dict[str, bool]] = None,
                  sampling_rate: Optional[float] = None) -> None:
        """Configure the tracer with custom settings.

        Args:
            flush_interval: How often to flush spans to the backend (seconds)
            max_spans: Maximum number of spans to buffer before forcing a flush
            collect_code_details: Whether to collect code details for spans
            integrations: Dict of integration names to enable/disable
            sampling_rate: Sampling rate from 0.0 to 1.0 (1.0 = sample everything)
        """
        if flush_interval is not None:
            self._flush_interval = flush_interval
            logger.info(f"Tracer flush_interval configured to {flush_interval}s.")
        if max_spans is not None:
            self._max_spans = max_spans
            logger.info(f"Tracer max_spans configured to {max_spans}.")
        if collect_code_details is not None:
            self.collect_code_details = collect_code_details
            logger.info(f"Tracer collect_code_details configured to {collect_code_details}.")
        if integrations is not None:
            # Note: this will not re-setup integrations if they are already active.
            # This configuration should ideally be called before integrations are used.
            self._integrations_config.update(integrations)
            logger.info(f"Tracer integrations configured to: {self._integrations_config}")
        if sampling_rate is not None:
            self._sampling_rate = max(0.0, min(1.0, sampling_rate))  # Clamp to [0, 1]
            logger.info(f"Tracer sampling_rate configured to {self._sampling_rate}.")

    def set_global_tags(self, tags: dict[str, str]) -> None:
        """Set global tags that are applied to all spans, traces, and sessions.

        These tags are merged into every new span's tags, and are also propagated
        to trace- and session-level tag stores so that they affect current and
        future spans within active contexts.
        """
        if not isinstance(tags, dict):
            raise TypeError("tags must be a dictionary")

        # Filter out None values to avoid null overwrites downstream
        filtered = {k: v for k, v in tags.items() if v is not None}
        if not filtered:
            return

        # Update global store
        self._global_tags.update(filtered)

        # Apply to any currently active spans (best-effort)
        stack = self._active_spans_ctx.get()
        for sp in stack:
            sp.tags.update(filtered)
            sp.trace_tags.update(filtered)
            sp.session_tags.update(filtered)

        # Ensure current trace/session contexts inherit these as well
        # so future spans in the same contexts get the tags too.
        active_trace_ids = {sp.trace_id for sp in stack if getattr(sp, "trace_id", None)}
        for tid in active_trace_ids:
            self._trace_level_tags.setdefault(tid, {}).update(filtered)
        active_session_ids = {sp.session_id for sp in stack if getattr(sp, "session_id", None)}
        for sid in active_session_ids:
            if sid:
                self._session_level_tags.setdefault(sid, {}).update(filtered)

    def start_span(
        self,
        name: str,
        kind: str = "generic",
        attributes: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        trace_tags: Optional[dict[str, str]] = None,
        session_tags: Optional[dict[str, str]] = None,
        is_new_trace: bool = False,
        trace_id: Optional[str] = None
    ) -> Span:
        """Start a new span; roots may create a session automatically."""
        # Ensure integrations are initialized before starting any spans
        self.ensure_integrations_initialized()

        if self.is_shutting_down():
            logger.warning("Tracer is shutting down. Discarding new span.")
            # Return a no-op span if tracer is shutting down
            return Span(name="noop_span", attributes={"warning": "Tracer is shutting down."})

        stack = self._active_spans_ctx.get()
        parent_span = stack[-1] if stack else None

        # Check for OpenTelemetry context if no ZeroEval parent
        otel_trace_id = None
        otel_parent_id = None
        otel_session_id = None
        otel_session_name = None

        if not parent_span and HAS_OTEL:
            otel_span = otel_trace.get_current_span()
            if otel_span and otel_span.is_recording():
                # Extract trace context from OpenTelemetry span
                span_context = otel_span.get_span_context()
                if span_context.is_valid:
                    # Convert OTEL IDs to UUID format
                    # OTEL trace ID is 128-bit, convert to UUID format
                    otel_trace_hex = format(span_context.trace_id, '032x')
                    otel_trace_id = str(uuid.UUID(otel_trace_hex))

                    # OTEL span ID is only 64-bit, we'll store as hex in parent_id
                    # but need to convert to UUID format for database
                    otel_span_hex = format(span_context.span_id, '016x')
                    # Create a deterministic UUID from the span ID
                    # Use the span ID as the first 16 chars, pad with zeros
                    padded_hex = otel_span_hex + '0' * 16
                    otel_parent_id = str(uuid.UUID(padded_hex))

                    is_new_trace = False  # We're continuing an existing OTEL trace

                    # Extract session information from OTEL span attributes or baggage
                    if hasattr(otel_span, 'attributes') and otel_span.attributes:
                        # Check for session ID in span attributes
                        # Common attribute names for session ID
                        session_attrs = [
                            'zeroeval.session.id', 'session.id', 'session_id',
                            'trace.session_id'
                        ]
                        for attr in session_attrs:
                            if attr in otel_span.attributes:
                                otel_session_id = str(otel_span.attributes[attr])
                                # Found session ID in OTEL attributes
                                break

                        # Check for session name
                        name_attrs = ['zeroeval.session.name', 'session.name', 'session_name']
                        for attr in name_attrs:
                            if attr in otel_span.attributes:
                                otel_session_name = str(otel_span.attributes[attr])
                                # Found session name in OTEL attributes
                                break

                    # If no session ID found, try to get it from baggage
                    if not otel_session_id and HAS_OTEL:
                        try:
                            from opentelemetry import baggage
                            ctx = otel_trace.get_current()
                            session_from_baggage = baggage.get_baggage('session_id', ctx)
                            if session_from_baggage:
                                otel_session_id = str(session_from_baggage)
                                # Found session ID in baggage
                        except Exception:
                            pass  # Could not extract baggage

                    # If still no session ID, check if we have a session for this trace ID
                    if not otel_session_id and otel_trace_id:
                        # Check if we've seen this trace before and have a session for it
                        existing_session = self._trace_to_session.get(otel_trace_id)
                        if existing_session:
                            otel_session_id = existing_session['id']
                            otel_session_name = existing_session.get('name')
                            # Found existing session for this trace

                    # Log for debugging
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Found OTEL context - trace_id: {otel_trace_id}, parent_id: {otel_parent_id}")

        # If is_new_trace is True, ignore parent span for creating a new trace
        if is_new_trace:
            parent_span = None

        # --- decide final session id and name -------------------------------------
        # Session IDs are normalized to UUIDs at creation time to ensure consistency
        # between what get_current_span() returns and what's stored in the database.
        if session_id:
            # Explicitly provided session ID takes precedence - normalize it
            normalized_id, original_as_name = _normalize_session_id(session_id)
            final_session_id = normalized_id
            # Use provided name, or fall back to original string if it was normalized
            final_session_name = session_name if session_name else original_as_name
        elif parent_span:
            # Inherit from parent span (already normalized)
            final_session_id = parent_span.session_id
            final_session_name = parent_span.session_name  # Inherit parent's name
        elif otel_session_id:
            # Use session ID from OTEL context if available - normalize it
            normalized_id, original_as_name = _normalize_session_id(otel_session_id)
            final_session_id = normalized_id
            final_session_name = otel_session_name or session_name or original_as_name
        else:
            # Create or reuse process-level session for root span without any context
            env_sid = os.environ.get("ZEROEVAL_SESSION_ID")
            if not env_sid:
                env_sid = str(uuid.uuid4())
                os.environ["ZEROEVAL_SESSION_ID"] = env_sid
            # Normalize env session ID as well
            normalized_id, original_as_name = _normalize_session_id(env_sid)
            final_session_id = normalized_id
            final_session_name = session_name or os.environ.get("ZEROEVAL_SESSION_NAME") or original_as_name
        # -----------------------------------------------------------------

        # Create new span
        span = Span(
            name=name,
            kind=kind,
            parent_id=parent_span.span_id if parent_span else otel_parent_id,
            attributes=attributes or {},
            session_id=final_session_id,
            session_name=final_session_name,
            tags=tags or {},
            trace_tags=trace_tags or {},
            session_tags=session_tags or {},
            ab_choices=parent_span.ab_choices.copy() if parent_span else []
        )

        # Apply global tags baseline (do not override explicitly provided keys)
        if self._global_tags:
            for k, v in self._global_tags.items():
                span.tags.setdefault(k, v)
                span.trace_tags.setdefault(k, v)
                span.session_tags.setdefault(k, v)

        logger.info(f"Starting span: {span.name} (new_trace={is_new_trace})")

        # Set trace ID based on context (priority: explicit > OTEL > parent > new)
        if trace_id:
            # Explicitly provided trace ID takes highest priority
            span.trace_id = trace_id
        elif otel_trace_id:
            # Use OTEL trace ID if we found one
            span.trace_id = otel_trace_id
            if attributes is None:
                span.attributes = {}
            span.attributes["otel.context"] = True
        elif parent_span and not is_new_trace:
            # Otherwise inherit from parent span
            span.trace_id = parent_span.trace_id

        # ---------------------------------------------------------------
        # Inherit/bubble tags from parent span, trace-level, session-level
        # ---------------------------------------------------------------
        if parent_span:
            inherited = parent_span.tags.copy()
            inherited.update(span.tags)  # Child overrides duplicates
            span.tags = inherited

        # Ensure span.trace_tags/session_tags include any previously registered tags
        if span.trace_id in self._trace_level_tags:
            inherited_tags = self._trace_level_tags[span.trace_id]
            span.trace_tags.update(inherited_tags)
            span.tags.update(inherited_tags)
        if span.session_id and span.session_id in self._session_level_tags:
            inherited_sess = self._session_level_tags[span.session_id]
            span.session_tags.update(inherited_sess)
            span.tags.update(inherited_sess)

        # Add this span to the stack for this thread
        self._active_spans_ctx.set(stack + [span])

        # --- Reference counting for the trace -------------------
        trace_id = span.trace_id
        if trace_id not in self._traces:
            # New trace: decide sampling unless coming from OTEL
            if otel_trace_id:
                # When joining existing OTEL trace, do not drop it
                self._traces[trace_id] = Trace(trace_id=trace_id, is_sampled=True)
            else:
                is_sampled = random.random() < self._sampling_rate
                self._traces[trace_id] = Trace(trace_id=trace_id, is_sampled=is_sampled)
                if not is_sampled and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Trace {trace_id} not sampled (sampling_rate={self._sampling_rate})")
        # Increment ref count for every started span in the trace
        self._traces[trace_id].ref_count += 1

        # Store trace-to-session mapping for OTEL context propagation
        if trace_id and span.session_id:
            self._trace_to_session[trace_id] = {
                'id': span.session_id,
                'name': span.session_name
            }
            # Mapped trace to session for OTEL context propagation

        return span

    def end_span(self, span: Span) -> None:
        """Ends the span, adds it to the buffer, and triggers a flush if needed."""
        if not span.end_time:
            span.end()

        if self.is_shutting_down() or span.name == "noop_span":
            return # Discard spans if shutting down or if it's a no-op span

        trace_id = span.trace_id

        # Always remove from active spans stack (even for unsampled traces)
        stack = self._active_spans_ctx.get()
        if stack and stack[-1].span_id == span.span_id:
            self._active_spans_ctx.set(stack[:-1])

        # Check sampling decision before cleanup
        is_sampled = True  # Default to sampled if trace not in registry
        if trace_id in self._traces:
            is_sampled = self._traces[trace_id].is_sampled

            # Decrement reference count for trace cleanup
            self._traces[trace_id].ref_count -= 1

            # Clean up trace if no more active spans
            if self._traces[trace_id].ref_count <= 0:
                logger.debug(f"Cleaning up trace {trace_id} (ref_count=0, sampled={is_sampled})")
                del self._traces[trace_id]
                # Also clean up trace-to-session mapping
                if trace_id in self._trace_to_session:
                    del self._trace_to_session[trace_id]

        # Check if this trace is being sampled - if not, skip buffering/sending
        if not is_sampled:
            logger.debug(f"Not sending span {span.name} from unsampled trace {trace_id}")
            return

        duration = span.duration_ms
        logger.info(
            f"Ending span: {span.name} (status: {span.status}, duration: {duration:.2f}ms)"
            if duration else f"Ending span: {span.name} (status: {span.status})"
        )

        with self._flush_lock:
            self._spans.append(span.to_dict())

            # Trigger flush if buffer is full
            if len(self._spans) >= self._max_spans:
                logger.info(f"Span buffer at max capacity ({self._max_spans}). Triggering flush.")
                self.flush(in_lock=True) # Already holding lock

    def flush(self, in_lock: bool = False) -> None:
        """
        Flush all buffered completed spans to the writer.
        The `in_lock` parameter prevents deadlocks when called from a context
        that already holds the flush lock.
        """
        def _do_flush():
            if not self._spans:
                logger.info("No spans to flush (buffer is empty).")
                return

            spans_to_flush = self._spans.copy()
            self._spans.clear()
            self._last_flush_time = time.time()

            logger.info(f"=== TRACER FLUSH: Starting flush of {len(spans_to_flush)} spans ===")

            # Log span summaries being flushed
            logger.info("Spans being flushed:")
            for i, span in enumerate(spans_to_flush):
                logger.info(f"  [{i+1}/{len(spans_to_flush)}] {span.get('name', 'unnamed')}")
                logger.info(f"       - span_id: {span.get('span_id')}")
                logger.info(f"       - trace_id: {span.get('trace_id')}")
                logger.info(f"       - session_id: {span.get('session_id')}")
                logger.info(f"       - status: {span.get('status')}")
                logger.info(f"       - duration_ms: {span.get('duration_ms')}")

            # Call writer and capture the result
            try:
                result = self.writer.write(spans_to_flush)

                # Log the writer result
                logger.info(f"=== TRACER FLUSH RESULT ===")
                if result:
                    if result.get("success"):
                        logger.info(f"✅ Flush SUCCESSFUL!")
                        logger.info(f"  - Spans sent: {result.get('spans_sent', len(spans_to_flush))}")
                        logger.info(f"  - Status code: {result.get('status_code', 'N/A')}")
                        logger.info(f"  - Response time: {result.get('response_time', 'N/A')}s")
                        if result.get('response_body'):
                            logger.info(f"  - Response preview: {result.get('response_body')[:200]}...")
                    else:
                        logger.error(f"❌ Flush FAILED!")
                        logger.error(f"  - Error: {result.get('error', 'Unknown error')}")
                        logger.error(f"  - Status code: {result.get('status_code', 'N/A')}")
                        logger.error(f"  - Exception type: {result.get('exception_type', 'N/A')}")
                        if result.get('response_body'):
                            logger.error(f"  - Error response: {result.get('response_body')}")
                else:
                    logger.warning("Writer returned no result information")

                logger.info(f"=== TRACER FLUSH COMPLETE ===")

            except Exception as e:
                logger.error(f"=== TRACER FLUSH EXCEPTION ===")
                logger.error(f"Unexpected error during flush: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {str(e)}", exc_info=True)

        if in_lock:
            _do_flush()
        else:
            with self._flush_lock:
                _do_flush()

    def _flush_periodically(self) -> None:
        """Background thread that flushes spans based on time interval."""
        while not self.is_shutting_down():
            time.sleep(self._flush_interval)

            # Check if it's time to flush based on interval
            with self._flush_lock:
                is_buffer_non_empty = bool(self._spans)
                time_since_last_flush = time.time() - self._last_flush_time

            if is_buffer_non_empty and time_since_last_flush >= self._flush_interval:
                logger.info(f"Periodic flush triggered after {self._flush_interval}s interval.")
                self.flush()

    def current_span(self) -> Optional[Span]:
        """Return the current active span for this thread, if any."""
        stack = self._active_spans_ctx.get()
        return stack[-1] if stack else None

    # ------------------------------------------------------------------
    # Tag helpers (trace / session level)
    # ------------------------------------------------------------------
    def add_trace_tags(self, trace_id: str, tags: dict[str, str]):
        """Attach *tags* to an entire trace (root + all children, past & future)."""
        if not tags:
            return
        self._trace_level_tags.setdefault(trace_id, {}).update(tags)
        # Update active spans only. The backend is responsible for back-filling
        # tags on already-flushed spans for the same trace.
        stack = self._active_spans_ctx.get()
        for sp in stack:
            if sp.trace_id == trace_id:
                sp.trace_tags.update(tags)
                sp.tags.update(tags)

    def add_session_tags(self, session_id: str, tags: dict[str, str]):
        """Attach *tags* to every span within a session."""
        if not tags:
            return

        # Filter out None values from tags
        filtered_tags = {k: v for k, v in tags.items() if v is not None}
        if not filtered_tags:
            return

        self._session_level_tags.setdefault(session_id, {}).update(filtered_tags)
        # Update active spans only.
        stack = self._active_spans_ctx.get()
        for sp in stack:
            if sp.session_id == session_id:
                sp.session_tags.update(filtered_tags)
                sp.tags.update(filtered_tags)

    def is_active_trace(self, trace_id: str) -> bool:
        """
        Return True if *trace_id* is known to the tracer by being present
        in an active (un-ended) span. This check is indicative, not exhaustive.
        """
        stack = self._active_spans_ctx.get()
        return any(sp.trace_id == trace_id for sp in stack)

    # ------------------------------------------------------------------
    # Convenience helper methods
    # ------------------------------------------------------------------
    def get_current_span(self) -> Optional[Span]:
        """Return the current active Span (or None)."""
        return self.current_span()

    def get_current_trace(self) -> Optional[str]:
        """Return the current trace_id (or None if no active span)."""
        current = self.current_span()
        return current.trace_id if current else None

    def get_current_session(self) -> Optional[str]:
        """Return the current session_id (or None if no active span)."""
        current = self.current_span()
        return current.session_id if current else None

    def set_tag(self, target, tags: dict[str, str]) -> None:
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
            if self.is_active_trace(target):
                self.add_trace_tags(target, tags)
            else:
                self.add_session_tags(target, tags)
        else:
            raise TypeError("Unsupported target type for set_tag")

    # ------------------------------------------------------------------
    # Convenience methods for current context
    # ------------------------------------------------------------------
    def set_current_tag(self, key: str, value: str) -> None:
        """Set a tag on the current span. If no span is active, this is a no-op."""
        current = self.current_span()
        if current:
            current.tags[key] = value
        else:
            logger.warning("No active span to set tag on")

    def set_trace_tag(self, key: str, value: str) -> None:
        """Set a tag on the current trace. If no trace is active, this is a no-op."""
        trace_id = self.get_current_trace()
        if trace_id:
            self.add_trace_tags(trace_id, {key: value})
        else:
            logger.warning("No active trace to set tag on")

    def set_session_tag(self, key: str, value: str) -> None:
        """Set a tag on the current session. If no session is active, this is a no-op."""
        session_id = self.get_current_session()
        if session_id:
            self.add_session_tags(session_id, {key: value})
        else:
            logger.warning("No active session to set tag on")


# Create the global tracer instance
tracer = Tracer()

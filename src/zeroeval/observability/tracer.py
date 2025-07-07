import atexit
import builtins
import contextlib
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .span import Span
from .writer import SpanBackendWriter, SpanWriter

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    """Represents a collection of spans and metadata for a single trace."""
    trace_id: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    ref_count: int = 0


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
        self._writer: SpanWriter = SpanBackendWriter()
        self._flush_interval: float = 1.0  # Flush more frequently for streaming
        self._max_spans: int = 20
        self._flush_lock = threading.Lock()
        self._integrations: dict[str, Any] = {}
        
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
        
        # Containers for trace- and session-level tags
        self._trace_level_tags: dict[str, dict[str, str]] = {}
        self._session_level_tags: dict[str, dict[str, str]] = {}
        
        logger.info("Initializing tracer for streaming...")
        logger.info(f"Tracer config: flush_interval={self._flush_interval}s, max_spans={self._max_spans}")

        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_periodically, daemon=True)
        self._flush_thread.start()

        # Auto-setup integrations
        self._setup_available_integrations()
        
        # Register shutdown hook
        atexit.register(self.shutdown)
    
    def _setup_available_integrations(self) -> None:
        """Automatically set up all available integrations."""
        # Import here to avoid circular imports
        from .integrations.langchain.integration import LangChainIntegration
        from .integrations.langgraph.integration import LangGraphIntegration
        from .integrations.openai.integration import OpenAIIntegration
        
        # List of all integration classes
        integration_classes = [
            OpenAIIntegration,
            LangChainIntegration,  # Auto-instrument LangChain
            LangGraphIntegration,  # Auto-instrument LangGraph
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
                  integrations: Optional[dict[str, bool]] = None) -> None:
        """Configure the tracer with custom settings."""
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
    
    def start_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        trace_tags: Optional[dict[str, str]] = None,
        session_tags: Optional[dict[str, str]] = None
    ) -> Span:
        """Start a new span; roots may create a session automatically."""
        if self.is_shutting_down():
            logger.warning("Tracer is shutting down. Discarding new span.")
            # Return a no-op span if tracer is shutting down
            return Span(name="noop_span", attributes={"warning": "Tracer is shutting down."})

        stack = self._active_spans_ctx.get()
        parent_span = stack[-1] if stack else None
        
        # --- decide final session id and name -------------------------------------
        if session_id:
            final_session_id = session_id
            final_session_name = session_name  # Use provided name or None
        elif parent_span:
            final_session_id = parent_span.session_id
            final_session_name = parent_span.session_name  # Inherit parent's name
        else:
            final_session_id = str(uuid.uuid4())    # new session for root
            final_session_name = session_name  # Use provided name or None
        # -----------------------------------------------------------------
        
        # Create new span
        span = Span(
            name=name,
            parent_id=parent_span.span_id if parent_span else None,
            attributes=attributes or {},
            session_id=final_session_id,
            session_name=final_session_name,
            tags=tags or {},
            trace_tags=trace_tags or {},
            session_tags=session_tags or {}
        )
        
        logger.info(f"Starting span: {span.name}")
        
        # If there's a parent span, both spans should share the same trace ID
        if parent_span:
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
            self._traces[trace_id] = Trace(trace_id=trace_id)
        self._traces[trace_id].ref_count += 1
        
        return span
    
    def end_span(self, span: Span) -> None:
        """Ends the span, adds it to the buffer, and triggers a flush if needed."""
        if not span.end_time:
            span.end()
            
        if self.is_shutting_down() or span.name == "noop_span":
            return # Discard spans if shutting down or if it's a no-op span

        duration = span.duration_ms
        logger.info(
            f"Ending span: {span.name} (status: {span.status}, duration: {duration:.2f}ms)"
            if duration else f"Ending span: {span.name} (status: {span.status})"
        )
        
        stack = self._active_spans_ctx.get()
        if stack and stack[-1].span_id == span.span_id:
            self._active_spans_ctx.set(stack[:-1])
        
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
                return

            spans_to_flush = self._spans.copy()
            self._spans.clear()
            self._last_flush_time = time.time()

            logger.info(f"Flushing {len(spans_to_flush)} spans to writer.")
            self._writer.write(spans_to_flush)
            
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
        self._session_level_tags.setdefault(session_id, {}).update(tags)
        # Update active spans only.
        stack = self._active_spans_ctx.get()
        for sp in stack:
            if sp.session_id == session_id:
                sp.session_tags.update(tags)
                sp.tags.update(tags)

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
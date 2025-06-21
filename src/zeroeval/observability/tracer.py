import threading
import time
import atexit
from typing import List, Dict, Any, Optional, Type
from .span import Span
from .writer import SpanWriter, SpanBackendWriter
import uuid
import logging
import atexit

logger = logging.getLogger(__name__)


class Tracer:
    """
    Singleton tracer that manages spans, buffering, and flushing to writers.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Tracer, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the tracer's internal state and register for graceful shutdown."""
        self._spans: List[Dict[str, Any]] = []
        self._active_spans: Dict[int, List[Span]] = {}
        self._trace_buckets: Dict[str, List[Dict[str, Any]]] = {}
        self._trace_counts: Dict[str, int] = {}
        self._last_flush_time = time.time()
        self._writer: SpanWriter = SpanBackendWriter()
        self._flush_interval: float = 10.0
        self._max_spans: int = 100
        self._flush_lock = threading.Lock()
        self._integrations: Dict[str, Any] = {}
        self.collect_code_details: bool = True
        self._shutdown_called: bool = False
        self._shutdown_lock = threading.Lock()
        
        logger.info("Initializing tracer...")
        logger.info(f"Tracer config: flush_interval={self._flush_interval}s, max_spans={self._max_spans}")

        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_periodically, daemon=True)
        self._flush_thread.start()

        # Auto-setup integrations immediately (reverted behavior)
        self._setup_available_integrations()
        
        # Register a single shutdown hook that will run on program exit.
        atexit.register(self.shutdown)
    
    def _setup_available_integrations(self) -> None:
        """Automatically set up all available integrations."""
        # Import here to avoid circular imports
        from .integrations.openai.integration import OpenAIIntegration
        from .integrations.langchain.integration import LangChainIntegration
        from .integrations.langgraph.integration import LangGraphIntegration
        
        # List of all integration classes
        integration_classes = [
            OpenAIIntegration,
            LangChainIntegration,  # Auto-instrument LangChain
            LangGraphIntegration,  # Auto-instrument LangGraph
        ]
        
        logger.info(f"Checking for available integrations: {[i.__name__ for i in integration_classes]}")
        # Setup each available integration
        for integration_class in integration_classes:
            if integration_class.is_available():
                try:
                    logger.info(f"Setting up integration: {integration_class.__name__}")
                    integration = integration_class(self)
                    integration.setup()
                    self._integrations[integration_class.__name__] = integration
                except Exception:
                    logger.error(f"Failed to set up integration {integration_class.__name__}", exc_info=True)
            else:
                logger.info(f"Integration not available: {integration_class.__name__}")
        
        if self._integrations:
            logger.info(f"Active integrations: {list(self._integrations.keys())}")
        else:
            logger.info("No active integrations found.")

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
        
        # Teardown integrations safely
        for integration_name, integration in self._integrations.items():
            try:
                integration.teardown()
            except Exception:
                logger.error(f"Failed to teardown integration {integration_name}", exc_info=True)

        # To ensure all traces are flushed, we must move any remaining buckets
        # to the main flush buffer, as they wouldn't be moved otherwise if their
        # root spans haven't technically ended before interpreter shutdown.
        with self._flush_lock:
            for trace_id, bucket in self._trace_buckets.items():
                logger.info(f"Flushing incomplete trace '{trace_id}' on shutdown.")
                bucket_sorted = sorted(bucket, key=lambda s: s.get('parent_id') is not None)
                self._spans.extend(bucket_sorted)

            # Clear the buckets to avoid duplicates if shutdown is called manually
            self._trace_buckets.clear()
            self._trace_counts.clear()

        self.flush()

    def __del__(self):
        """Cleanup integrations when tracer is destroyed."""
        for integration in self._integrations.values():
            try:
                integration.teardown()
            except:
                pass
    
    def configure(self, 
                  flush_interval: Optional[float] = None,
                  max_spans: Optional[int] = None,
                  collect_code_details: Optional[bool] = None) -> None:
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
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        session_name: Optional[str] = None
    ) -> Span:
        """Start a new span; roots may create a session automatically."""
        if self.is_shutting_down():
            logger.warning("Tracer is shutting down. Discarding new span.")
            # Return a no-op span if tracer is shutting down
            return Span(name="noop_span", attributes={"warning": "Tracer is shutting down."})

        thread_id = threading.get_ident()
        
        # Initialize span stack for this thread if it doesn't exist
        if thread_id not in self._active_spans:
            self._active_spans[thread_id] = []
        
        # Get parent span if available
        parent_span = self._active_spans[thread_id][-1] if self._active_spans[thread_id] else None
        
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
            session_name=final_session_name
        )
        
        logger.info(f"Starting span: {span.name}")
        
        # If there's a parent span, both spans should share the same trace ID
        if parent_span:
            span.trace_id = parent_span.trace_id
        
        # Add this span to the stack for this thread
        self._active_spans[thread_id].append(span)
        
        # --- Reference counting for the trace -------------------
        self._trace_counts[span.trace_id] = self._trace_counts.get(span.trace_id, 0) + 1
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End the given span, add it to the trace bucket, and handle trace completion."""
        if not span.end_time:
            span.end()
            
        if self.is_shutting_down() or span.name == "noop_span":
            return # Discard spans if shutting down or if it's a no-op span

        duration = span.duration_ms
        logger.info(
            f"Ending span: {span.name} (status: {span.status}, duration: {duration:.2f}ms)"
            if duration else f"Ending span: {span.name} (status: {span.status})"
        )
        
        thread_id = threading.get_ident()
        
        # Remove from active spans if it's the current one for this thread
        if self._active_spans.get(thread_id) and self._active_spans[thread_id][-1].span_id == span.span_id:
            self._active_spans[thread_id].pop()
            if not self._active_spans[thread_id]:
                del self._active_spans[thread_id]
        
        # Add to bucket and check for trace completion
        trace_id = span.trace_id
        if trace_id not in self._trace_buckets:
            self._trace_buckets[trace_id] = []
        self._trace_buckets[trace_id].append(span.to_dict())

        self._trace_counts[trace_id] -= 1
        if self._trace_counts[trace_id] == 0:
            self._finalize_trace(trace_id)

    def _finalize_trace(self, trace_id: str):
        """Sort, buffer, and clean up a completed trace."""
        bucket = self._trace_buckets.pop(trace_id, [])
        if not bucket:
            return

        # Sort the bucket to ensure the root span is first
        bucket_sorted = sorted(bucket, key=lambda s: s.get('parent_id') is not None)

        with self._flush_lock:
            self._spans.extend(bucket_sorted)

        # Cleanup
        self._trace_counts.pop(trace_id, None)

        # Trigger flush if buffer is full
        if len(self._spans) >= self._max_spans:
            logger.info(f"Span buffer at max capacity ({self._max_spans}). Triggering flush.")
            self.flush()
    
    def flush(self) -> None:
        """Flush all buffered completed spans to the writer."""
        with self._flush_lock:
            if not self._spans:
                return

            spans_to_flush = self._spans.copy()
            self._spans.clear()
            self._last_flush_time = time.time()

            logger.info(f"Flushing {len(spans_to_flush)} spans to writer.")
            self._writer.write(spans_to_flush)
    
    def _flush_periodically(self) -> None:
        """Background thread that flushes spans based on time interval."""
        while True:
            time.sleep(1.0)  # Check every second
            current_time = time.time()
            if (current_time - self._last_flush_time) >= self._flush_interval:
                if self._spans:
                    logger.info(f"Periodic flush triggered after {self._flush_interval}s interval.")
                    self.flush()


# Create the global tracer instance
tracer = Tracer()
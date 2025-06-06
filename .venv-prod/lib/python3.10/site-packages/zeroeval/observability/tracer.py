import threading
import time
from typing import List, Dict, Any, Optional, Type
from .span import Span
from .writer import SpanWriter, ConsoleWriter, SpanBackendWriter
import uuid


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
        """Initialize the tracer's internal state."""
        self._spans: List[Dict[str, Any]] = []
        self._active_spans: Dict[int, List[Span]] = {}
        self._last_flush_time = time.time()
        self._writer: SpanWriter = SpanBackendWriter()
        self._flush_interval: float = 10.0
        self._max_spans: int = 100
        self._flush_lock = threading.Lock()
        self._integrations: Dict[str, Any] = {}
        
        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_periodically, daemon=True)
        self._flush_thread.start()
        
        # Auto-setup available integrations
        self._setup_available_integrations()
    
    def _setup_available_integrations(self) -> None:
        """Automatically set up all available integrations."""
        # Import here to avoid circular imports
        from .integrations.openai.integration import OpenAIIntegration
        
        # List of all integration classes
        integration_classes = [
            OpenAIIntegration,
            # Add new integration classes here
        ]
        
        # Setup each available integration
        for integration_class in integration_classes:
            if integration_class.is_available():
                try:
                    integration = integration_class(self)
                    integration.setup()
                    self._integrations[integration_class.__name__] = integration
                except Exception:
                    # Silently fail if integration setup fails
                    pass

    def __del__(self):
        """Cleanup integrations when tracer is destroyed."""
        for integration in self._integrations.values():
            try:
                integration.teardown()
            except:
                pass
    
    def configure(self, 
                  flush_interval: Optional[float] = None,
                  max_spans: Optional[int] = None) -> None:
        """Configure the tracer with custom settings."""
        if flush_interval is not None:
            self._flush_interval = flush_interval
        if max_spans is not None:
            self._max_spans = max_spans
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Span:
        """Start a new span; roots may create a session automatically."""
        thread_id = threading.get_ident()
        
        # Initialize span stack for this thread if it doesn't exist
        if thread_id not in self._active_spans:
            self._active_spans[thread_id] = []
        
        # Get parent span if available
        parent_span = self._active_spans[thread_id][-1] if self._active_spans[thread_id] else None
        
        # --- decide final session id -------------------------------------
        if session_id:
            final_session_id = session_id
        elif parent_span:
            final_session_id = parent_span.session_id
        else:
            final_session_id = str(uuid.uuid4())    # new session for root
        # -----------------------------------------------------------------
        
        # Create new span
        span = Span(
            name=name,
            parent_id=parent_span.span_id if parent_span else None,
            attributes=attributes or {},
            session_id=final_session_id
        )
        
        # If there's a parent span, both spans should share the same trace ID
        if parent_span:
            span.trace_id = parent_span.trace_id
        
        # Add this span to the stack for this thread
        self._active_spans[thread_id].append(span)
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End the given span and remove it from active spans if it's the current one."""
        if not span.end_time:
            span.end()
        
        thread_id = threading.get_ident()
        
        # Only remove from active spans if it's on the stack
        if thread_id in self._active_spans and self._active_spans[thread_id]:
            # Check if this is the top span in the stack
            if self._active_spans[thread_id][-1].span_id == span.span_id:
                # Remove from the stack
                self._active_spans[thread_id].pop()
                
                # Remove the thread entry if there are no more spans
                if not self._active_spans[thread_id]:
                    self._active_spans.pop(thread_id, None)
        
        # Add to buffer
        self._spans.append(span.to_dict())
        
        # Check if we should flush
        if len(self._spans) >= self._max_spans:
            self.flush()
    
    def flush(self) -> None:
        """Flush all buffered spans to the writer."""
        with self._flush_lock:
            if not self._spans:
                return
            
            spans_to_flush = self._spans.copy()
            self._spans.clear()
            self._last_flush_time = time.time()
            
            # Write spans to the configured writer
            self._writer.write(spans_to_flush)
    
    def _flush_periodically(self) -> None:
        """Background thread that flushes spans based on time interval."""
        while True:
            time.sleep(1.0)  # Check every second
            current_time = time.time()
            if (current_time - self._last_flush_time) >= self._flush_interval:
                self.flush()


# Create the global tracer instance
tracer = Tracer()
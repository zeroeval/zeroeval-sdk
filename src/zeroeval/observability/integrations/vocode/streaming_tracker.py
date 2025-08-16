"""
Streaming tracker for Vocode operations.

Provides a clean way to track TTFB and duration for streaming operations
without requiring granular instrumentation.
"""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StreamingSpanTracker:
    """
    Wraps a streaming generator to track TTFB and total duration.
    Updates the span with timing metrics as the stream progresses.
    """
    
    def __init__(self, 
                 generator: AsyncGenerator,
                 span: Any,
                 tracer: Any,
                 operation_type: str = "streaming"):
        """
        Initialize the streaming tracker.
        
        Args:
            generator: The async generator to wrap
            span: The ZeroEval span to update
            tracer: The ZeroEval tracer
            operation_type: Type of operation (tts, stt, etc.)
        """
        self.generator = generator
        self.span = span
        self.tracer = tracer
        self.operation_type = operation_type
        self.start_time = time.perf_counter()
        self.first_chunk_time: Optional[float] = None
        self.chunk_count = 0
        self.total_bytes = 0
        self.finished = False
    
    async def __aiter__(self):
        """Async iteration that tracks timing."""
        try:
            async for item in self.generator:
                # Track first chunk (TTFB)
                if self.first_chunk_time is None:
                    self.first_chunk_time = time.perf_counter()
                    ttfb_ms = (self.first_chunk_time - self.start_time) * 1000
                    
                    # Update span with TTFB
                    self.span.attributes[f"{self.operation_type}.ttfb_ms"] = round(ttfb_ms, 1)
                    logger.debug(f"[{self.operation_type}] TTFB: {ttfb_ms:.1f}ms")
                
                # Track chunk metrics
                self.chunk_count += 1
                if hasattr(item, 'chunk') and item.chunk:
                    self.total_bytes += len(item.chunk)
                
                yield item
                
        finally:
            # Calculate total duration when stream ends
            if not self.finished:
                self.finished = True
                total_duration_ms = (time.perf_counter() - self.start_time) * 1000
                
                # Update span with final metrics
                self.span.attributes[f"{self.operation_type}.duration_ms"] = round(total_duration_ms, 1)
                self.span.attributes[f"{self.operation_type}.chunk_count"] = self.chunk_count
                
                if self.total_bytes > 0:
                    self.span.attributes[f"{self.operation_type}.total_bytes"] = self.total_bytes
                
                # Calculate streaming duration (time after first chunk)
                if self.first_chunk_time:
                    streaming_duration_ms = (time.perf_counter() - self.first_chunk_time) * 1000
                    self.span.attributes[f"{self.operation_type}.streaming_duration_ms"] = round(streaming_duration_ms, 1)
                
                logger.debug(
                    f"[{self.operation_type}] Completed: "
                    f"{self.chunk_count} chunks, "
                    f"{total_duration_ms:.1f}ms total"
                )
                
                # End the span
                self.tracer.end_span(self.span)


class SynthesisResultWrapper:
    """
    Wraps a Vocode SynthesisResult to track TTS streaming metrics.
    Proxies all attributes to the original result while tracking streaming.
    """
    
    def __init__(self, synthesis_result: Any, span: Any, tracer: Any, text: str):
        """
        Initialize the wrapper.
        
        Args:
            synthesis_result: The original SynthesisResult from Vocode
            span: The ZeroEval span for this TTS operation
            tracer: The ZeroEval tracer
            text: The text being synthesized
        """
        # Store original result for attribute proxying
        self._synthesis_result = synthesis_result
        self._span = span
        self._tracer = tracer
        self._text = text
        
        # Set input/output on span
        self._span.set_io(
            input_data=text,
            output_data=f"Audio stream ({len(text)} chars)"
        )
        
        # Wrap the chunk generator with tracking
        self.chunk_generator = StreamingSpanTracker(
            synthesis_result.chunk_generator,
            span,
            tracer,
            operation_type="tts"
        )
        
        # Explicitly copy key attributes
        self.get_message_up_to = synthesis_result.get_message_up_to
        if hasattr(synthesis_result, 'cached'):
            self.cached = synthesis_result.cached
            span.attributes["tts.cached"] = synthesis_result.cached
    
    def __getattr__(self, name):
        """
        Proxy any unknown attributes to the original synthesis result.
        This ensures compatibility with Vocode's expectations.
        """
        # Avoid infinite recursion when accessing _synthesis_result
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Proxy to the original synthesis result
        return getattr(self._synthesis_result, name)
    
    def __setattr__(self, name, value):
        """
        Handle attribute setting properly.
        """
        # Set our own private attributes
        if name.startswith('_') or name in ('chunk_generator', 'get_message_up_to', 'cached'):
            object.__setattr__(self, name, value)
        else:
            # Proxy to the original synthesis result
            if hasattr(self, '_synthesis_result'):
                setattr(self._synthesis_result, name, value)
            else:
                object.__setattr__(self, name, value)


def track_tts_streaming(span: Any, tracer: Any, text: str) -> Callable:
    """
    Returns a decorator that wraps TTS results with streaming tracking.
    
    Args:
        span: The ZeroEval span for the TTS operation
        tracer: The ZeroEval tracer  
        text: The text being synthesized
    
    Returns:
        A function that wraps SynthesisResult objects
    """
    def wrap_result(synthesis_result):
        """Wrap the synthesis result with tracking."""
        if synthesis_result is None:
            tracer.end_span(span)
            return synthesis_result
        
        return SynthesisResultWrapper(synthesis_result, span, tracer, text)
    
    return wrap_result

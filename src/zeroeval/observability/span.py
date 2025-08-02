import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union


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


@dataclass
class Span:
    """
    Represents a traced operation with OpenTelemetry-compatible attributes.
    """
    # Required fields first
    name: str
    
    # Optional fields with defaults
    kind: str = "generic"  # Type of span: generic, llm, tts, http, database, vector_store, etc.
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # Fields for tracking execution
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    code: Optional[str] = None  # Added code field
    code_filepath: Optional[str] = None
    code_lineno: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    status: str = "ok"
    tags: dict[str, str] = field(default_factory=dict)
    # Optional tags that should be applied to the owning trace and/or session when this
    # span is ingested. These will be processed by the backend ingestion service.
    trace_tags: dict[str, str] = field(default_factory=dict)
    session_tags: dict[str, str] = field(default_factory=dict)
    # Signals attached to this span
    signals: dict[str, Any] = field(default_factory=dict)

    def end(self) -> None:
        """Mark the span as completed with the current timestamp."""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get the span duration in milliseconds, if completed."""
        if self.end_time is None:
            return None
        
        return (self.end_time - self.start_time) * 1000
    
    def set_error(self, code: str, message: str, stack: Optional[str] = None) -> None:
        """Set error information for the span."""
        self.error_code = code
        self.error_message = message
        self.error_stack = stack
        self.status = 'error'

    def set_io(self, input_data: Optional[str] = None, output_data: Optional[str] = None) -> None:
        """Set input/output data for the span, without overwriting existing values."""
        if input_data is not None:
            self.input_data = input_data
        if output_data is not None:
            self.output_data = output_data

    def set_code(self, code: str) -> None:
        """Set the code that was executed in this span."""
        self.code = code

    def set_code_context(self, filepath: str, lineno: int) -> None:
        """Set the file path and line number for the span's execution context."""
        self.code_filepath = filepath
        self.code_lineno = lineno

    def set_signal(self, name: str, value: Union[str, bool, int, float]) -> None:
        """Set a signal for this span."""
        self.signals[name] = Signal(name=name, value=value)

    def to_dict(self) -> dict[str, Any]:
        """Convert the span to a dictionary representation."""
        # Convert signals to a serializable format
        signals_dict = {}
        for name, signal in self.signals.items():
            if hasattr(signal, 'value') and hasattr(signal, 'signal_type'):
                signals_dict[name] = {
                    'name': signal.name,
                    'value': signal.value,
                    'type': signal.signal_type
                }
            else:
                # Handle legacy signal format or simple values
                signals_dict[name] = signal
        
        span_dict = {
            "name": self.name,
            "kind": self.kind,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "kind": self.kind,  # Added kind field
            "attributes": self.attributes,
            "tags": self.tags,
            "trace_tags": self.trace_tags,
            "session_tags": self.session_tags,
            "signals": signals_dict,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "code": self.code,  # Added code field
            "code_filepath": self.code_filepath,
            "code_lineno": self.code_lineno,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "status": self.status
        }
        
        return span_dict

    def end(self, error: Optional[Exception] = None) -> None:
        """End the span, calculating duration and capturing errors."""
        if self.end_time:
            return  # Span already ended
            
        self.end_time = time.time()

        if error:
            self.status = "error"
            self.error_code = type(error).__name__
            self.error_message = str(error)
            self.error_stack = "".join(traceback.format_exception(error))
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass
class Span:
    """
    Represents a traced operation with OpenTelemetry-compatible attributes.
    """
    # Required fields first
    name: str
    
    # Optional fields with defaults
    session_id: Optional[str] = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Fields for tracking execution
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    code: Optional[str] = None  # Added code field
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    status: str = "ok"

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
        """Set input/output data for the span."""
        self.input_data = input_data
        self.output_data = output_data

    def set_code(self, code: str) -> None:
        """Set the code that was executed in this span."""
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        """Convert the span to a dictionary representation."""
        return {
            "name": self.name,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "code": self.code,  # Added code field
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "status": self.status
        }
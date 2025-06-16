import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone


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
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
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

    def end(self) -> None:
        """Mark the span as completed with the current timestamp."""
        self.end_time = datetime.now(timezone.utc).isoformat()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get the span duration in milliseconds, if completed."""
        if self.end_time is None:
            return None
        
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        
        return (end - start).total_seconds() * 1000
    
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

    def set_code_context(self, filepath: str, lineno: int) -> None:
        """Set the file path and line number for the span's execution context."""
        self.code_filepath = filepath
        self.code_lineno = lineno

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
            "code_filepath": self.code_filepath,
            "code_lineno": self.code_lineno,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_stack": self.error_stack,
            "status": self.status
        }
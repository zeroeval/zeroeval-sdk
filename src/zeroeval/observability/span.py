import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List


@dataclass
class Span:
    """
    Represents a traced operation with OpenTelemetry-compatible attributes.
    """
    name: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def end(self) -> None:
        """Mark the span as completed with the current timestamp."""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get the span duration in milliseconds, if completed."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the span to a dictionary representation."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes
        }
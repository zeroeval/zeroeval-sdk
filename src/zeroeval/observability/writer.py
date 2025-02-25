from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json

class SpanWriter(ABC):
    """Interface for writing spans to different destinations."""
    
    @abstractmethod
    def write(self, spans: List[Dict[str, Any]]) -> None:
        """Write a batch of spans to the destination."""
        pass


class ConsoleWriter(SpanWriter):
    """Writes spans to the console for debugging."""
    
    def write(self, spans: List[Dict[str, Any]]) -> None:
        """Print spans to the console in a readable format."""
        for span in spans:
            formatted_span = {
                "name": span["name"],
                "trace_id": span["trace_id"][:8] + "...",  # Shortened for readability
                "span_id": span["span_id"][:8] + "...",
                "parent_id": span["parent_id"][:8] + "..." if span["parent_id"] else None,
                "duration_ms": span["duration_ms"],
                "attributes": span["attributes"]
            }
            print(f"SPAN: {json.dumps(formatted_span, indent=2)}")
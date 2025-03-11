from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
import requests
from datetime import datetime

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
            short_trace = span["trace_id"][:8] + "..." if span["trace_id"] else None
            short_span = span["span_id"][:8] + "..." if span["span_id"] else None
            short_parent = (
                span["parent_id"][:8] + "..." if span["parent_id"] else None
            )
            formatted_span = {
                "name": span["name"],
                "trace_id": short_trace,
                "span_id": short_span,
                "parent_id": short_parent,
                "duration_ms": span["duration_ms"],
                "attributes": span["attributes"],
            }
            print(f"SPAN: {json.dumps(formatted_span, indent=2)}")


class SpanBackendWriter(SpanWriter):
    """
    A writer that sends spans to the backend for permanent storage.
    Assumes that your '/spans' route requires API key authentication.
    """

    def __init__(self) -> None:
        """Initialize the writer with an API URL and optional API key."""
        self.api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
        self.api_key = os.environ.get("API_KEY", "")

    def write(self, spans: List[Dict[str, Any]]) -> None:
        """
        Write a batch of spans to the '/spans' endpoint, ensuring the payload
        matches the backend's expected schema.
        """
        if not spans:
            return

        formatted_spans = []
        for span in spans:
            started_at_iso = (
                datetime.fromtimestamp(span["start_time"]).isoformat()
                if span["start_time"]
                else None
            )
            ended_at_iso = (
                datetime.fromtimestamp(span["end_time"]).isoformat()
                if span["end_time"]
                else None
            )

            formatted_span = {
                "id": span["span_id"],
                "trace_id": span["trace_id"],
                "parent_span_id": span["parent_id"],
                "name": span["name"],
                "started_at": started_at_iso,
                "ended_at": ended_at_iso,
                "duration_ms": span["duration_ms"],
                "attributes": span["attributes"],
                "status": span["status"],
                "input_data": span["input_data"],
                "output_data": span["output_data"],
                "error_code": span["error_code"],
                "error_message": span["error_message"],
                "error_stack": span["error_stack"],
                "experiment_result_id": "ece6859c-4a35-4a5b-b2f5-7c6a5e3863c8"
            }
            formatted_spans.append(formatted_span)

        endpoint = f"{self.api_url}/spans"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["X-API-KEY"] = self.api_key

        try:
            # Debug print in case you want to inspect the final payload
            print(formatted_spans)
            response = requests.post(endpoint, headers=headers, json=formatted_spans, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[SpanBackendWriter] Error posting spans: {exc}")
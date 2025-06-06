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
                "session_id": span.get("session_id"),
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
        # Default to production API; override in dev with API_URL env var (e.g., "https://api.zeroeval.com")
        self.api_url = os.environ.get("API_URL", "https://api.zeroeval.com").rstrip("/")

    def _get_api_key(self) -> str:
        """Get the API key from environment, supporting lazy loading after ze.init()."""
        return os.environ.get("API_KEY", "")

    def write(self, spans: List[Dict[str, Any]]) -> None:
        """
        Write a batch of spans to the '/spans' endpoint, ensuring the payload
        matches the backend's expected schema.
        """
        if not spans:
            return

        # Get API key at write time (after ze.init() has been called)
        api_key = self._get_api_key()
        
        # Debug logging
        print(f"[SpanBackendWriter] API_URL: {self.api_url}")
        print(f"[SpanBackendWriter] API_KEY present: {bool(api_key)}")
        if api_key:
            print(f"[SpanBackendWriter] API_KEY (first 8 chars): {api_key[:8]}...")
        else:
            print(f"[SpanBackendWriter] WARNING: No API_KEY found in environment!")

        formatted_spans = []
        for span in spans:
            try:
                # Convert traceback object to string if present
                error_stack = str(span.get("error_stack")) if span.get("error_stack") else None
                
                formatted_span = {
                    "id": span["span_id"],
                    "session_id": span.get("session_id"),
                    "trace_id": span["trace_id"],
                    "parent_span_id": span["parent_id"],
                    "name": span["name"],
                    "started_at": datetime.fromtimestamp(span["start_time"]).isoformat() if span.get("start_time") else None,
                    "ended_at": datetime.fromtimestamp(span["end_time"]).isoformat() if span.get("end_time") else None,
                    "duration_ms": span["duration_ms"],
                    "attributes": span.get("attributes", {}),
                    "status": span.get("status", "unset"),
                    "input_data": json.dumps(span["input_data"]) if isinstance(span["input_data"], (dict, list)) else span["input_data"],
                    "output_data": json.dumps(span["output_data"]) if isinstance(span["output_data"], (dict, list)) else span["output_data"],
                    "code": span.get("code"),
                    "error_code": span.get("error_code"),
                    "error_message": span.get("error_message"),
                    "error_stack": error_stack,
                    "experiment_result_id": span.get("experiment_result_id")
                }
                formatted_spans.append(formatted_span)
            except Exception as e:
                print(f"[SpanBackendWriter] Error formatting span: {e}")
                print(f"Problematic span: {span}")
                continue

        endpoint = f"{self.api_url}/spans/"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            print(f"[SpanBackendWriter] Using Bearer token authentication")
        else:
            print(f"[SpanBackendWriter] WARNING: No API key available, sending unauthenticated request")

        print(f"[SpanBackendWriter] Sending {len(formatted_spans)} spans to {endpoint}")
        print(f"[SpanBackendWriter] Headers: {dict(headers)}")

        try:
            print(formatted_spans)
            response = requests.post(endpoint, headers=headers, json=formatted_spans, timeout=10)
            print(f"[SpanBackendWriter] Response status: {response.status_code}")
            if not response.ok:
                print(f"[SpanBackendWriter] Error posting spans: Status {response.status_code}")
                print(f"Response: {response.text}")
                return
            response.raise_for_status()
            print(f"[SpanBackendWriter] Successfully posted {len(formatted_spans)} spans")
        except requests.RequestException as exc:
            print(f"[SpanBackendWriter] Error posting spans: {exc}")
            print(f"Request URL: {endpoint}")
            print(f"Headers: {headers}")
            if hasattr(exc, 'response') and exc.response:
                print(f"Response status: {exc.response.status_code}")
                print(f"Response body: {exc.response.text}")
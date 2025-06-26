from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
import os
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SpanWriter(ABC):
    """Interface for writing spans to different destinations."""
    
    @abstractmethod
    def write(self, spans: List[Dict[str, Any]]) -> None:
        """Write a batch of spans to the destination."""
        pass


class SpanBackendWriter(SpanWriter):
    """
    A writer that sends spans to the backend for permanent storage.
    Assumes that your '/spans' route requires API key authentication.
    """

    def __init__(self) -> None:
        """Initialize the writer with an API URL and optional API key."""
        self.api_url = os.environ.get("ZEROEVAL_API_URL", "https://api.zeroeval.com").rstrip("/")

    def _get_api_key(self) -> str:
        """Get the API key from environment, supporting lazy loading after ze.init()."""
        return os.environ.get("ZEROEVAL_API_KEY", "")

    def write(self, spans: List[Dict[str, Any]]) -> None:
        """
        Write a batch of spans to the '/spans' endpoint, ensuring the payload
        matches the backend's expected schema.
        """
        if not spans:
            return

        # Get API key at write time (after ze.init() has been called)
        api_key = self._get_api_key()

        formatted_spans = []
        for span in spans:
            try:
                # Convert traceback object to string if present
                error_stack = str(span.get("error_stack")) if span.get("error_stack") else None
                
                # Prepare session data if session_name is provided
                session_data = None
                if span.get("session_id"):
                    if span.get("session_name") or span.get("session_tags"):
                        session_data = {
                            "id": span["session_id"],
                            "name": span.get("session_name"),
                        }
                        # Attach tags if provided
                        if span.get("session_tags"):
                            session_data["tags"] = span["session_tags"]
                
                formatted_span = {
                    "id": span["span_id"],
                    "session_id": span.get("session_id"),
                    "trace_id": span["trace_id"],
                    "parent_span_id": span["parent_id"],
                    "name": span["name"],
                    "started_at": span.get("start_time"),
                    "ended_at": span.get("end_time"),
                    "duration_ms": span["duration_ms"],
                    "attributes": span.get("attributes", {}),
                    "status": span.get("status", "unset"),
                    "input_data": json.dumps(span["input_data"]) if isinstance(span["input_data"], (dict, list)) else span["input_data"],
                    "output_data": json.dumps(span["output_data"]) if isinstance(span["output_data"], (dict, list)) else span["output_data"],
                    "code": span.get("code"),
                    "code_filepath": span.get("code_filepath"),
                    "code_lineno": span.get("code_lineno"),
                    "error_code": span.get("error_code"),
                    "error_message": span.get("error_message"),
                    "error_stack": error_stack,
                    "experiment_result_id": span.get("experiment_result_id"),
                    "tags": span.get("tags", {}),
                    "trace_tags": span.get("trace_tags", {}),
                    "session_tags": span.get("session_tags", {})
                }
                
                # Add session object if we have session name
                if session_data:
                    formatted_span["session"] = session_data
                formatted_spans.append(formatted_span)
            except Exception:
                logger.error(f"Failed to format span: {span.get('name', 'unnamed')}", exc_info=True)
                continue

        if not formatted_spans:
            logger.info("No spans to write after formatting.")
            return

        endpoint = f"{self.api_url}/spans"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        logger.info(f"Sending {len(formatted_spans)} spans to {endpoint}")
        # Debug: log the first span to see what's being sent
        if formatted_spans:
            logger.info(f"First span being sent: {json.dumps(formatted_spans[0], indent=2)}")
        try:
            response = requests.post(endpoint, headers=headers, json=formatted_spans, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully posted {len(formatted_spans)} spans. Response: {response.status_code}")
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(
                    "Authorization error sending traces. Please check your API key, "
                    "set via `ze.init(api_key=...)` or the `ZEROEVAL_API_KEY` env variable. Generate a new key with `zeroeval setup`."
                )
            else:
                logger.error(
                    f"Error posting spans to {endpoint}: HTTP {e.response.status_code} "
                    f"Response: {e.response.text}"
                )
        except requests.RequestException as e:
            logger.error(f"Error posting spans to {endpoint}: {e}")
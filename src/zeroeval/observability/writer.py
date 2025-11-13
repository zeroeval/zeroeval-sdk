import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import requests
import uuid

from .utils import safe_json_dumps, sanitize_for_json

logger = logging.getLogger(__name__)

class SpanWriter(ABC):
    """Interface for writing spans to different destinations."""
    
    @abstractmethod
    def write(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
        """Write a batch of spans to the destination.
        
        Returns:
            A dictionary with write result information including success status,
            response details, and any error information.
        """
        pass


class SpanBackendWriter(SpanWriter):
    """
    A writer that sends spans to the backend for permanent storage.
    Assumes that your '/spans' route requires API key authentication.
    """

    def __init__(self) -> None:
        """Initialize the writer with an API URL and optional API key."""
        # Don't read API URL at init time - read it lazily at write time
        # This allows ze.init() to set the environment variable first
        pass

    def _get_api_url(self) -> str:
        """Get the API URL from environment, supporting lazy loading after ze.init()."""
        return os.environ.get("ZEROEVAL_API_URL", "https://api.zeroeval.com").rstrip("/")

    def _get_api_key(self) -> str:
        """Get the API key from environment, supporting lazy loading after ze.init()."""
        return os.environ.get("ZEROEVAL_API_KEY", "")

    def write(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Write a batch of spans to the '/spans' endpoint, ensuring the payload
        matches the backend's expected schema. This writer is intentionally simple,
        sending spans as they are received from the tracer's buffer.
        """
        if not spans:
            return {"success": True, "spans_sent": 0, "message": "No spans to write"}

        # Get API URL and key at write time
        api_url = self._get_api_url()
        api_key = self._get_api_key()

        # Define filter_null_values function before its first use
        def filter_null_values(d):
            """Remove keys with None/null values from a dictionary."""
            if not d:
                return {}
            return {k: v for k, v in d.items() if v is not None}

        # Sort so that for each session, the first span with a non-empty session name appears
        # before any spans without a name.  This makes the backend upsert receive the proper
        # name even when multiple spans share the same session in the same flush batch.
        spans_sorted = sorted(
            spans,
            key=lambda s: (s.get("session_id"), bool((s.get("session_name") or "").strip())),
            reverse=True,
        )

        formatted_spans = []
        for span in spans_sorted:
            try:
                # Sanitize the entire span to ensure all fields are JSON serializable
                sanitized_span = sanitize_for_json(span)
                # DEBUG: Log session information for this span
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Formatting span '%s' (session_id=%s, session_name=%s)",
                        sanitized_span.get("name"),
                        sanitized_span.get("session_id"),
                        repr(sanitized_span.get("session_name")),
                    )

                # Convert traceback object to string if present
                error_stack = str(sanitized_span.get("error_stack")) if sanitized_span.get("error_stack") else None
                
                # Prepare session data, normalizing session_id to UUID when necessary
                session_data = None
                session_id_val = sanitized_span.get("session_id")
                session_name_val = sanitized_span.get("session_name")
                session_tags = sanitized_span.get("session_tags")

                # Helper to detect UUID-like strings (36-char with hyphens)
                def _looks_like_uuid(val: Any) -> bool:
                    try:
                        uuid.UUID(str(val))
                        return True
                    except Exception:
                        return False

                normalized_session_id = None
                if session_id_val:
                    if _looks_like_uuid(session_id_val):
                        normalized_session_id = str(session_id_val)
                    else:
                        # Deterministically derive a UUIDv5 from the provided session_id string
                        normalized_session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(session_id_val)))
                        # If no explicit name provided, use the original string as the name
                        if session_name_val is None:
                            session_name_val = str(session_id_val)

                # Only include session object if we have a normalized ID and either a name or tags
                if normalized_session_id and (session_name_val is not None or session_tags):
                    session_data = {"id": normalized_session_id}
                    if session_name_val is not None:
                        session_data["name"] = session_name_val
                    if session_tags:
                        session_data["tags"] = filter_null_values(session_tags)

                # DEBUG: Log whether session_data will be included
                if logger.isEnabledFor(logging.DEBUG):
                    if session_data:
                        logger.debug(
                            "Session object to be sent for session_id %s: %s",
                            sanitized_span.get("session_id"),
                            session_data,
                        )
                    else:
                        logger.debug(
                            "No session metadata sent for session_id %s (no name/tags)",
                            sanitized_span.get("session_id"),
                        )
                
                formatted_span = {
                    "id": sanitized_span["span_id"],
                    "session_id": normalized_session_id or sanitized_span.get("session_id"),
                    "trace_id": sanitized_span["trace_id"],
                    "parent_span_id": sanitized_span["parent_id"],
                    "name": sanitized_span["name"],
                    "kind": sanitized_span.get("kind") or sanitized_span.get("attributes", {}).get("kind", "generic"),  # Check attributes as fallback
                    "started_at": sanitized_span.get("start_time"),
                    "ended_at": sanitized_span.get("end_time"),
                    "duration_ms": sanitized_span["duration_ms"],
                    "attributes": sanitized_span.get("attributes", {}),
                    "status": sanitized_span.get("status", "unset"),
                    "input_data": safe_json_dumps(sanitized_span["input_data"]) if isinstance(sanitized_span["input_data"], (dict, list)) else sanitized_span["input_data"],
                    "output_data": safe_json_dumps(sanitized_span["output_data"]) if isinstance(sanitized_span["output_data"], (dict, list)) else sanitized_span["output_data"],
                    "code": sanitized_span.get("code"),
                    "code_filepath": sanitized_span.get("code_filepath"),
                    "code_lineno": sanitized_span.get("code_lineno"),
                    "error_code": sanitized_span.get("error_code"),
                    "error_message": sanitized_span.get("error_message"),
                    "error_stack": error_stack,
                    "experiment_result_id": sanitized_span.get("experiment_result_id"),
                    "tags": filter_null_values(sanitized_span.get("tags", {})),
                    "trace_tags": filter_null_values(sanitized_span.get("trace_tags", {})),
                    "session_tags": filter_null_values(sanitized_span.get("session_tags", {})),
                    "ab_choices": sanitized_span.get("ab_choices", [])
                }
                
                # Add session object if we have session name or tags
                if session_data:
                    formatted_span["session"] = session_data
                formatted_spans.append(formatted_span)
            except Exception:
                logger.error(
                    "Failed to format span during write: %s",
                    sanitized_span.get("name", "unnamed") if 'sanitized_span' in locals() else span.get("name", "unnamed"),
                    exc_info=True,
                )
                continue

        if not formatted_spans:
            logger.info("No spans to write after formatting.")
            return {"success": True, "spans_sent": 0, "message": "No spans to write after formatting"}

        endpoint = f"{api_url}/spans"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Log detailed information about each span being sent
        logger.info(f"=== ZEROEVAL API WRITE: Preparing to send {len(formatted_spans)} spans to {endpoint} ===")
        
        # Create a summary of spans being sent
        span_details = []
        for span in formatted_spans:
            detail = {
                "span_id": span["id"],
                "name": span["name"],
                "trace_id": span["trace_id"],
                "session_id": span.get("session_id"),
                "session_name": span.get("session", {}).get("name") if span.get("session") else None,
                "parent_span_id": span.get("parent_span_id"),
                "status": span.get("status"),
                "duration_ms": span.get("duration_ms"),
            }
            span_details.append(detail)
            
        # Log span details
        logger.info("Span details being sent:")
        for i, detail in enumerate(span_details):
            logger.info(f"  [{i+1}/{len(span_details)}] Span: {detail['name']}")
            logger.info(f"       - span_id: {detail['span_id']}")
            logger.info(f"       - trace_id: {detail['trace_id']}")
            logger.info(f"       - session_id: {detail['session_id']}")
            logger.info(f"       - session_name: {detail['session_name']}")
            logger.info(f"       - parent_span_id: {detail['parent_span_id']}")
            logger.info(f"       - status: {detail['status']}")
            logger.info(f"       - duration_ms: {detail['duration_ms']}")
        
        # Log request details
        logger.info("Request details:")
        logger.info(f"  - URL: {endpoint}")
        logger.info(f"  - Method: POST")
        logger.info(f"  - Headers: {headers}")
        logger.info(f"  - API Key Present: {'Yes' if api_key else 'No'}")
        logger.info(f"  - Payload size: {len(safe_json_dumps(formatted_spans))} bytes")
        
        # Log full payload at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full request payload: {safe_json_dumps(formatted_spans, indent=2)}")
        
        try:
            logger.info("Sending request...")
            response = requests.post(endpoint, headers=headers, json=formatted_spans, timeout=10)
            
            # Log response details regardless of status
            logger.info(f"=== ZEROEVAL API RESPONSE ===")
            logger.info(f"  - Status Code: {response.status_code}")
            logger.info(f"  - Status Text: {response.reason}")
            logger.info(f"  - Headers: {dict(response.headers)}")
            logger.info(f"  - Response Time: {response.elapsed.total_seconds():.3f}s")
            
            # Log response body
            try:
                response_body = response.json()
                logger.info(f"  - Response Body: {safe_json_dumps(response_body, indent=2)}")
            except:
                logger.info(f"  - Response Body (text): {response.text}")
            
            response.raise_for_status()
            
            logger.info(f"✅ Successfully posted {len(formatted_spans)} spans!")
            logger.info(f"=== ZEROEVAL API WRITE COMPLETE ===")
            
            # Return success information for the tracer to log
            return {
                "success": True,
                "status_code": response.status_code,
                "spans_sent": len(formatted_spans),
                "response_time": response.elapsed.total_seconds(),
                "response_body": response.text[:500] if response.text else None  # First 500 chars
            }
            
        except requests.HTTPError as e:
            logger.error(f"=== ZEROEVAL API ERROR RESPONSE ===")
            logger.error(f"  - Status Code: {e.response.status_code}")
            logger.error(f"  - Status Text: {e.response.reason}")
            logger.error(f"  - Headers: {dict(e.response.headers)}")
            
            # Log error response body
            try:
                error_body = e.response.json()
                logger.error(f"  - Error Response Body: {safe_json_dumps(error_body, indent=2)}")
            except:
                logger.error(f"  - Error Response Body (text): {e.response.text}")
            
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
            
            # Return error information
            return {
                "success": False,
                "status_code": e.response.status_code,
                "error": str(e),
                "response_body": e.response.text[:500] if e.response.text else None
            }
            
        except requests.RequestException as e:
            logger.error(f"=== ZEROEVAL API REQUEST ERROR ===")
            logger.error(f"Error posting spans to {endpoint}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            
            # Return error information
            return {
                "success": False,
                "error": str(e),
                "exception_type": type(e).__name__
            }
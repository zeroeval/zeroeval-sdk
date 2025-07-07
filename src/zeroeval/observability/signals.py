import logging
import os
from typing import Union

import requests

from .span import Span
from .tracer import tracer

logger = logging.getLogger(__name__)


def _send_signals_immediately(
    entity_type: str, entity_id: str, signals: dict[str, Union[str, bool, int, float]]
) -> bool:
    """
    Private helper to send a batch of signals for a single entity immediately.
    """
    # Get configuration from environment
    api_url = os.getenv("ZEROEVAL_API_URL", "http://localhost:8000")
    api_key = os.getenv("ZEROEVAL_API_KEY")
    workspace_id = os.getenv("ZEROEVAL_WORKSPACE_ID")

    if not all([api_key, workspace_id, api_url]):
        logger.warning(
            "Cannot send signals. Missing ZEROEVAL_API_KEY, ZEROEVAL_WORKSPACE_ID, or ZEROEVAL_API_URL."
        )
        return False

    # Prepare payload
    endpoint = f"{api_url}/signals/bulk"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    api_payloads = [
        {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "name": name,
            "value": value,
            "signal_type": "numerical"
            if isinstance(value, (int, float))
            else "boolean",
        }
        for name, value in signals.items()
    ]
    payload = {"signals": api_payloads}

    # Send immediately and log status
    try:
        logger.info(
            f"Sending {len(signals)} signals for {entity_type}:{entity_id} immediately..."
        )
        response = requests.post(endpoint, json=payload, headers=headers, timeout=5.0)
        logger.info(
            f"Response for {entity_type}:{entity_id}: HTTP {response.status_code}"
        )
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send signals for {entity_type}:{entity_id}: {e}")
        return False


def set_signal(
    target: Union[Span, str], signals: dict[str, Union[str, bool, int, float]]
) -> bool:
    """
    Send signals immediately for a given span, trace, or session.

    This is a fire-and-forget operation that sends signals directly to the
    ZeroEval backend, independent of the span flushing mechanism.

    Args:
        target: The entity to attach signals to. Can be a `Span` object,
                a `trace_id` string, or a `session_id` string.
        signals: A dictionary of signal names to values.

    Returns:
        True if the signals were sent successfully, False otherwise.
    """
    if not isinstance(signals, dict) or not signals:
        logger.warning("No signals provided, nothing to send.")
        return True

    # Determine entity type and ID from the target
    entity_type = "session"  # Default assumption
    entity_id = None

    if isinstance(target, Span):
        entity_type = "span"
        entity_id = target.span_id
    elif isinstance(target, str):
        entity_id = target
        if tracer.is_active_trace(target):
            entity_type = "trace"
    else:
        raise TypeError(
            f"Unsupported target type '{type(target).__name__}' for signal. Must be Span or str."
        )

    return _send_signals_immediately(entity_type, entity_id, signals)

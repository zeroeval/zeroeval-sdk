import logging
import os
import random
from typing import Dict, Any, Union

import requests

from .span import Span
from .tracer import tracer

logger = logging.getLogger(__name__)

# Cache for choices made within the same context to ensure consistency
_choice_cache: Dict[str, str] = {}


def choose(
    name: str,
    variants: Dict[str, Any],
    weights: Dict[str, float]
) -> Any:
    """
    Make an A/B test choice using weighted random selection.

    This function automatically attaches the choice to the current span, trace, or session
    context. Choices are cached per entity to ensure consistency within the same context.

    Args:
        name: Name of the choice/test (e.g., "model_selection", "ui_variant")
        variants: Dictionary mapping variant keys to their values
                 e.g., {"a": "gpt-4", "b": "claude-3"}
        weights: Dictionary mapping variant keys to their selection weights
                e.g., {"a": 0.3, "b": 0.7}

    Returns:
        The value from the selected variant

    Example:
        model = ze.choose(
            "model_test",
            variants={"a": "gpt-4", "b": "claude-3"},
            weights={"a": 0.3, "b": 0.7}
        )
        # Returns either "gpt-4" (30% chance) or "claude-3" (70% chance)

    Raises:
        ValueError: If variants and weights don't have matching keys
        TypeError: If unable to determine current context
    """
    if not variants:
        raise ValueError("variants dictionary cannot be empty")

    if not weights:
        raise ValueError("weights dictionary cannot be empty")

    # Validate that variants and weights have matching keys
    variant_keys = set(variants.keys())
    weight_keys = set(weights.keys())

    if variant_keys != weight_keys:
        raise ValueError(
            f"Variant keys {variant_keys} must match weight keys {weight_keys}"
        )

    # Validate that weights sum to a reasonable value (allow some floating point tolerance)
    weight_sum = sum(weights.values())
    if not (0.95 <= weight_sum <= 1.05):
        logger.warning(
            f"Weights for choice '{name}' sum to {weight_sum:.3f}, not 1.0. "
            "This may cause unexpected probability distributions."
        )

    # Determine current context (similar to signals)
    current_span = tracer.get_current_span()
    current_trace = tracer.get_current_trace()
    current_session = tracer.get_current_session()

    entity_type = None
    entity_id = None

    # Prioritize span, then trace, then session (most specific to least specific)
    if current_span:
        entity_type = "span"
        entity_id = current_span.span_id
    elif current_trace:
        entity_type = "trace"
        entity_id = current_trace
    elif current_session:
        entity_type = "session"
        entity_id = current_session
    else:
        raise RuntimeError(
            "ze.choose() must be called within an active span, trace, or session context. "
            "Make sure you're calling it within a @ze.span decorated function or ze.span() context manager."
        )

    # Create cache key for this entity + choice combination
    cache_key = f"{entity_type}:{entity_id}:{name}"

    # Check if we've already made this choice for this entity
    if cache_key in _choice_cache:
        selected_key = _choice_cache[cache_key]
        logger.debug(
            f"Using cached choice for {name}: {selected_key} -> {variants[selected_key]}"
        )
        return variants[selected_key]

    # Make weighted random selection
    variant_keys_list = list(variants.keys())
    variant_weights = [weights[key] for key in variant_keys_list]

    selected_key = random.choices(variant_keys_list, weights=variant_weights, k=1)[0]
    selected_value = variants[selected_key]

    # Cache the choice
    _choice_cache[cache_key] = selected_key

    logger.info(
        f"Made choice '{name}' for {entity_type}:{entity_id}: "
        f"{selected_key} -> {selected_value}"
    )

    # Send choice data to backend (fire-and-forget)
    try:
        _send_choice_data(
            entity_type=entity_type,
            entity_id=entity_id,
            choice_name=name,
            variant_key=selected_key,
            variant_value=str(selected_value)
        )
    except Exception as e:
        logger.warning(f"Failed to send choice data for {name}: {e}")
        # Don't raise - choice selection should still work even if logging fails

    return selected_value


def _send_choice_data(
    entity_type: str,
    entity_id: str,
    choice_name: str,
    variant_key: str,
    variant_value: str
) -> bool:
    """
    Send choice data to the backend immediately.

    Returns:
        True if successful, False otherwise
    """
    # Get configuration from environment
    api_url = os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com")
    api_key = os.getenv("ZEROEVAL_API_KEY")

    if not all([api_key, api_url]):
        logger.warning(
            "Cannot send choice data. Missing ZEROEVAL_API_KEY or ZEROEVAL_API_URL."
        )
        return False

    # Prepare payload
    endpoint = f"{api_url}/ab-choices"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "choice_name": choice_name,
        "variant_key": variant_key,
        "variant_value": variant_value
    }

    try:
        logger.debug(f"Sending choice data for {choice_name} to {endpoint}")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=5.0)
        response.raise_for_status()
        logger.debug(f"Choice data sent successfully: HTTP {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send choice data: {e}")
        return False


def clear_choice_cache() -> None:
    """
    Clear the choice cache.

    This is mainly useful for testing or when you want to force new choices
    to be made for the same entities.
    """
    global _choice_cache
    _choice_cache.clear()
    logger.debug("Choice cache cleared")
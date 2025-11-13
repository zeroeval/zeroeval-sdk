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
    weights: Dict[str, float],
    duration_days: int,
    default_variant: str | None = None
) -> Any:
    """
    Make an A/B test choice using weighted random selection with experiment timeboxing.

    This function automatically attaches the choice to the current span, trace, or session
    context. Choices are cached per entity to ensure consistency within the same context.
    
    The experiment runs for the specified duration_days, after which the backend automatically
    stops accepting new choices. This ensures experiments are timebound and conclusions can be
    drawn from a fixed observation window.

    Args:
        name: Name of the choice/test (e.g., "model_selection", "ui_variant")
        variants: Dictionary mapping variant keys to their values
                 e.g., {"control": "gpt-4", "variant_a": "claude-3"}
        weights: Dictionary mapping variant keys to their selection weights
                e.g., {"control": 0.5, "variant_a": 0.5}
        duration_days: How many days the experiment should run (required for timeboxing)
        default_variant: Fallback variant key to use if the test has completed
                        (defaults to first variant key if not specified)

    Returns:
        The value from the selected variant

    Example:
        model = ze.choose(
            name="model_test",
            variants={"control": "gpt-4", "variant_a": "claude-3"},
            weights={"control": 0.5, "variant_a": 0.5},
            duration_days=14,
            default_variant="control"
        )
        # Returns either "gpt-4" or "claude-3" for 14 days, then defaults to "gpt-4"

    Note:
        Use ze.set_signal() to attach boolean success/failure signals to the same entity
        to enable signal-based analytics in the ZeroEval dashboard.

    Raises:
        ValueError: If variants and weights don't have matching keys, or duration_days <= 0
        RuntimeError: If unable to determine current context
    """
    if not variants:
        raise ValueError("variants dictionary cannot be empty")

    if not weights:
        raise ValueError("weights dictionary cannot be empty")

    if duration_days <= 0:
        raise ValueError("duration_days must be greater than 0")

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

    # Set default variant to first key if not specified
    if default_variant is None:
        default_variant = list(variants.keys())[0]
    elif default_variant not in variants:
        raise ValueError(f"default_variant '{default_variant}' not found in variants")

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

    logger.info(
        f"Made choice '{name}' for {entity_type}:{entity_id}: "
        f"{selected_key} -> {selected_value}"
    )

    # Send choice data to backend
    try:
        response_data = _send_choice_data(
            entity_type=entity_type,
            entity_id=entity_id,
            choice_name=name,
            variant_key=selected_key,
            variant_value=str(selected_value),
            variants=variants,
            weights=weights,
            duration_days=duration_days
        )
        
        # Check if test has completed or been manually ended
        if response_data and response_data.get("test_status") == "completed":
            logger.warning(
                f"A/B test '{name}' has completed. Using default variant '{default_variant}'. "
                f"Message: {response_data.get('message', 'Test ended')}"
            )
            # Cache the default variant for consistency within this context
            _choice_cache[cache_key] = default_variant
            return variants[default_variant]
        
        # Test is running - cache the random selection
        _choice_cache[cache_key] = selected_key
        
        # Attach AB choice to the current span for linkage
        if response_data and response_data.get("ab_choice_id") and current_span:
            ab_choice_metadata = {
                "ab_choice_id": response_data["ab_choice_id"],
                "choice_name": name,
                "variant_key": selected_key,
                "variant_value": str(selected_value)
            }
            current_span.ab_choices.append(ab_choice_metadata)
            logger.debug(f"Attached AB choice {response_data['ab_choice_id']} to span {current_span.span_id}")
            
    except Exception as e:
        logger.warning(f"Failed to send choice data for {name}: {e}")
        # Cache the selection even if API call failed to ensure consistency
        _choice_cache[cache_key] = selected_key
        # Don't raise - choice selection should still work even if logging fails

    return selected_value


def _send_choice_data(
    entity_type: str,
    entity_id: str,
    choice_name: str,
    variant_key: str,
    variant_value: str,
    variants: Dict[str, Any],
    weights: Dict[str, float],
    duration_days: int
) -> Dict[str, Any] | None:
    """
    Send choice data to the backend immediately.

    Returns:
        Response data dict if successful, None otherwise
    """
    # Get configuration from environment
    api_url = os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com")
    api_key = os.getenv("ZEROEVAL_API_KEY")

    if not all([api_key, api_url]):
        logger.warning(
            "Cannot send choice data. Missing ZEROEVAL_API_KEY or ZEROEVAL_API_URL."
        )
        return None

    # Prepare payload with new fields
    endpoint = f"{api_url}/ab-choices"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Convert variants to string values for serialization
    serialized_variants = {k: str(v) for k, v in variants.items()}

    payload = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "choice_name": choice_name,
        "variant_key": variant_key,
        "variant_value": variant_value,
        "variants": serialized_variants,
        "weights": weights,
        "duration_days": duration_days
    }

    try:
        logger.debug(f"Sending choice data for {choice_name} to {endpoint}")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=5.0)
        response.raise_for_status()
        logger.debug(f"Choice data sent successfully: HTTP {response.status_code}")
        
        # Parse and return response
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send choice data: {e}")
        return None


def clear_choice_cache() -> None:
    """
    Clear the choice cache.

    This is mainly useful for testing or when you want to force new choices
    to be made for the same entities.
    """
    global _choice_cache
    _choice_cache.clear()
    logger.debug("Choice cache cleared")
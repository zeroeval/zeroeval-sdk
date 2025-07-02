import os
import logging
from typing import Dict, Union, Any
import requests

logger = logging.getLogger(__name__)


def send_signal(entity_type: str, entity_id: str, signals: Dict[str, Union[str, bool, int, float]]) -> bool:
    """
    Send signals immediately to the ZeroEval backend.
    
    Args:
        entity_type: Type of entity (span, trace, session, completion)
        entity_id: UUID of the entity
        signals: Dictionary of signal name->value pairs
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get configuration
    api_url = os.getenv("ZEROEVAL_API_URL", "http://localhost:8000")
    api_key = os.getenv("ZEROEVAL_API_KEY")
    workspace_id = os.getenv("ZEROEVAL_WORKSPACE_ID")
    
    if not api_key:
        logger.warning("⚠️  ZEROEVAL_API_KEY not set, cannot send signals")
        return False
        
    if not workspace_id:
        logger.warning("⚠️  ZEROEVAL_WORKSPACE_ID not set, cannot send signals")
        return False
    
    # Build endpoint URL
    endpoint = f"{api_url}/workspaces/{workspace_id}/signals/bulk"
    
    # Convert signals to API format
    signal_payloads = []
    for name, value in signals.items():
        # Determine signal type
        signal_type = "numerical" if isinstance(value, (int, float)) else "boolean"
        
        signal_payloads.append({
            "entity_type": entity_type,
            "entity_id": entity_id,
            "name": name,
            "value": value,
            "signal_type": signal_type
        })
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {"signals": signal_payloads}
    
    # Send immediately
    try:
        logger.info(f"🌐 Sending {len(signals)} signals for {entity_type}:{entity_id}")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=5.0)
        
        # Log the result
        if response.status_code == 201:
            logger.info(f"✅ Signals sent successfully (HTTP {response.status_code})")
            try:
                result = response.json()
                if result.get("failed_count", 0) > 0:
                    logger.warning(f"⚠️  Some signals failed: {result.get('failed_count')} failures")
            except:
                pass  # Response might not be JSON, that's ok
            return True
        else:
            logger.error(f"❌ Failed to send signals (HTTP {response.status_code}): {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ Timeout sending signals (5s)")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error sending signals: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error sending signals: {e}")
        return False


def signal(target, signals: Dict[str, Union[str, bool, int, float]]) -> bool:
    """
    Send signals immediately to any entity (span, trace, session).
    
    This is the main function users should call. It follows the same pattern 
    as ze.set_tag() but sends signals immediately to the backend.
    
    Args:
        target: Can be a Span object, trace_id string, or session_id string
        signals: Dictionary of signal name->value pairs
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        # Send to current span
        current_span = ze.get_current_span()
        ze.signal(current_span, {"user_thumbs_up": True})
        
        # Send to trace/session by ID
        ze.signal("trace-id-123", {"overall_quality": 0.95})
        ze.signal("session-id-456", {"conversion_occurred": True})
    """
    if not isinstance(signals, dict):
        raise TypeError("signals must be a dictionary")
    
    if not signals:
        logger.warning("No signals provided, nothing to send")
        return True
    
    # Determine entity type and ID
    if hasattr(target, 'span_id'):
        # It's a Span object
        entity_type = "span"
        entity_id = target.span_id
    elif isinstance(target, str):
        # It's a string ID - we need to guess if it's trace or session
        # For now, we'll assume it's a session ID
        # In a more sophisticated implementation, we could check format or ask tracer
        entity_type = "session"  # Default assumption
        entity_id = target
        
        # Try to be smarter about trace vs session
        # If tracer knows about this as an active trace, use that
        try:
            from .tracer import tracer
            if tracer.is_active_trace(target):
                entity_type = "trace"
        except:
            pass  # Fallback to session is fine
    else:
        raise TypeError("target must be a Span object or string ID")
    
    logger.info(f"📤 Sending signals {list(signals.keys())} to {entity_type}: {entity_id}")
    return send_signal(entity_type, entity_id, signals) 
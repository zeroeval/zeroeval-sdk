"""Utility functions for the observability module."""

import json
import uuid
from datetime import datetime, date
from decimal import Decimal
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles non-serializable types gracefully.
    
    This encoder converts common non-serializable types to their string representations
    to prevent serialization errors during span flushing.
    """
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable formats."""
        try:
            # Handle UUID objects
            if isinstance(obj, uuid.UUID):
                return str(obj)
            
            # Handle datetime objects
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            
            # Handle Decimal objects
            elif isinstance(obj, Decimal):
                return float(obj)
            
            # Handle bytes
            elif isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except UnicodeDecodeError:
                    # If not valid UTF-8, encode as base64
                    import base64
                    return base64.b64encode(obj).decode('utf-8')
            
            # Handle sets by converting to lists
            elif isinstance(obj, set):
                return list(obj)
            
            # Handle any object with a __dict__ attribute
            elif hasattr(obj, '__dict__'):
                return {
                    '_type': obj.__class__.__name__,
                    '_module': obj.__class__.__module__,
                    'data': obj.__dict__
                }
            
            # Handle any object with a __str__ method
            elif hasattr(obj, '__str__'):
                return str(obj)
            
            # Let the base class raise the TypeError
            return super().default(obj)
            
        except Exception as e:
            # If all else fails, return a string representation
            logger.warning(f"Failed to serialize object of type {type(obj).__name__}: {e}")
            return f"<{type(obj).__name__} object>"


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON, handling non-serializable types.
    
    Args:
        obj: The object to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation of the object
    """
    # Use our custom encoder by default
    kwargs.setdefault('cls', SafeJSONEncoder)
    return json.dumps(obj, **kwargs)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to ensure it's JSON serializable.
    
    This function walks through nested structures and converts non-serializable
    objects to serializable formats.
    
    Args:
        obj: The object to sanitize
        
    Returns:
        A JSON-serializable version of the object
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    elif isinstance(obj, Decimal):
        return float(obj)
    
    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            import base64
            return base64.b64encode(obj).decode('utf-8')
    
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    elif isinstance(obj, set):
        return [sanitize_for_json(item) for item in obj]
    
    elif hasattr(obj, '__dict__'):
        return {
            '_type': obj.__class__.__name__,
            '_module': obj.__class__.__module__,
            'data': sanitize_for_json(obj.__dict__)
        }
    
    else:
        # For any other type, convert to string
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__} object>"

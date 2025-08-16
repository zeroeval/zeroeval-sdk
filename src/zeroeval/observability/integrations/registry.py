"""
Integration registry for lazy loading.

This module provides a registry system for integrations that avoids importing
all integration modules at startup, reducing initialization overhead.
"""

from typing import Dict, Type, Callable, Optional
import importlib
import logging

from .base import Integration

logger = logging.getLogger(__name__)


# Registry of integration name -> (module_path, class_name, package_name)
INTEGRATION_REGISTRY: Dict[str, tuple[str, str, str]] = {
    "OpenAIIntegration": (
        "zeroeval.observability.integrations.openai.integration",
        "OpenAIIntegration",
        "openai"
    ),
    "LangChainIntegration": (
        "zeroeval.observability.integrations.langchain.integration",
        "LangChainIntegration",
        "langchain_core"
    ),
    "LangGraphIntegration": (
        "zeroeval.observability.integrations.langgraph.integration",
        "LangGraphIntegration",
        "langgraph"
    ),
    "GeminiIntegration": (
        "zeroeval.observability.integrations.gemini.integration",
        "GeminiIntegration",
        "google.genai"
    ),
    "HttpxIntegration": (
        "zeroeval.observability.integrations.httpx.integration",
        "HttpxIntegration",
        "httpx"
    ),
    "VocodeIntegration": (
        "zeroeval.observability.integrations.vocode.integration",
        "VocodeIntegration",
        "vocode"
    ),

}


def is_package_available(package_name: str) -> bool:
    """Check if a package is available without importing it fully."""
    try:
        # Use importlib.util to check without actually importing
        import importlib.util
        spec = importlib.util.find_spec(package_name.split('.')[0])
        return spec is not None
    except (ImportError, AttributeError, ValueError):
        return False


def get_integration_class(integration_name: str) -> Optional[Type[Integration]]:
    """
    Lazily load and return an integration class.
    
    This avoids importing integration modules until they're actually needed.
    """
    if integration_name not in INTEGRATION_REGISTRY:
        logger.warning(f"Unknown integration: {integration_name}")
        return None
        
    module_path, class_name, package_name = INTEGRATION_REGISTRY[integration_name]
    
    # First check if the required package is available
    if not is_package_available(package_name):
        logger.debug(f"Package {package_name} not available for {integration_name}")
        return None
    
    try:
        # Only import the integration module if the package is available
        module = importlib.import_module(module_path)
        integration_class = getattr(module, class_name)
        return integration_class
    except Exception as e:
        logger.debug(f"Failed to load integration {integration_name}: {e}")
        return None


def get_available_integrations() -> Dict[str, Type[Integration]]:
    """
    Get all available integration classes.
    
    This lazily checks and loads only the integrations whose packages are installed.
    """
    available = {}
    for integration_name in INTEGRATION_REGISTRY:
        integration_class = get_integration_class(integration_name)
        if integration_class:
            available[integration_name] = integration_class
    return available
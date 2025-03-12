from typing import Type, Dict
from .base import Integration

def discover_integrations() -> Dict[str, Type[Integration]]:
    """
    Discover all available integrations.
    This can be expanded to use entry points for plugin-style discovery.
    """
    from .openai.integration import OpenAIIntegration
    
    return {
        "openai": OpenAIIntegration,
    }
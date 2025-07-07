from typing import Dict, Type

from .base import Integration


def discover_integrations() -> Dict[str, Type[Integration]]:
    """
    Discover all available integrations.
    This can be expanded to use entry points for plugin-style discovery.
    """
    from .langchain.integration import LangChainIntegration
    from .langgraph.integration import LangGraphIntegration
    from .openai.integration import OpenAIIntegration

    return {
        "openai": OpenAIIntegration,
        "langchain": LangChainIntegration,
        "langgraph": LangGraphIntegration,
    }


from .base import Integration


def discover_integrations() -> dict[str, type[Integration]]:
    """
    Discover all available integrations.
    This can be expanded to use entry points for plugin-style discovery.
    """
    from .langchain.integration import LangChainIntegration
    from .langgraph.integration import LangGraphIntegration
    from .openai.integration import OpenAIIntegration
    from .livekit.integration import LiveKitIntegration
    
    return {
        "openai": OpenAIIntegration,
        "langchain": LangChainIntegration,
        "langgraph": LangGraphIntegration,
        "livekit": LiveKitIntegration,
    }
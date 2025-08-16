import logging
import os
import uuid
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """A custom formatter to add colors to log levels."""

    grey = "\x1b[38;5;244m"
    blue = "\x1b[34;1m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, datefmt=None):
        super().__init__(datefmt=datefmt)
        self.FORMATS = {
            logging.DEBUG: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.blue}[%(levelname)s]{self.reset} %(message)s",
            logging.INFO: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.green}[%(levelname)s]{self.reset} %(message)s",
            logging.WARNING: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.yellow}[%(levelname)s]{self.reset} %(message)s",
            logging.ERROR: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.red}[%(levelname)s]{self.reset} %(message)s",
            logging.CRITICAL: f"{self.grey}[%(asctime)s]{self.reset} {self.blue}[%(name)s]{self.reset} {self.bold_red}[%(levelname)s]{self.reset} %(message)s",
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


def init(
    api_key: str = None, 
    workspace_name: str = "Personal Workspace",
    debug: bool = False,
    api_url: str = "https://api.zeroeval.com",
    disabled_integrations: list[str] = None,
    enabled_integrations: list[str] = None,
    setup_otlp: bool = True,
    service_name: str = "zeroeval-app",
    tags: Optional[dict[str, str]] = None
):
    """
    Initialize the ZeroEval SDK.
    
    Args:
        api_key (str, optional): Your ZeroEval API key.
        workspace_name (str, optional): The name of your workspace.
        debug (bool, optional): If True, enables detailed logging for debugging. 
                                Can also be enabled by setting the ZEROEVAL_DEBUG=true 
                                environment variable.
        api_url (str, optional): The URL of the ZeroEval API.
        disabled_integrations (list[str], optional): List of integrations to disable.
                                Use lowercase names: 'openai', 'gemini', 'langchain', 'langgraph'
                                Use this when you have compatibility issues with automatic patching.
        enabled_integrations (list[str], optional): List of integrations to enable.
                                If provided, ONLY these integrations will be loaded.
                                Use lowercase names: 'openai', 'gemini', 'langchain', 'langgraph'
                                This can significantly reduce startup time by avoiding unnecessary imports.
        setup_otlp (bool, optional): If True, sets up OpenTelemetry tracer provider for OTLP export.
                                This enables LiveKit and other OTLP-compatible libraries to send
                                spans to ZeroEval. Defaults to True.
        service_name (str, optional): Service name for OTLP traces. Defaults to "zeroeval-app".
        tags (dict[str, str], optional): Global tags to apply to all traces, sessions, and spans.
    """
    # Set workspace name (always use the provided value)
    os.environ["ZEROEVAL_WORKSPACE_NAME"] = workspace_name
    
    # Only override environment variables if values are explicitly provided
    if api_key is not None:
        os.environ["ZEROEVAL_API_KEY"] = api_key
    if api_url is not None:
        os.environ["ZEROEVAL_API_URL"] = api_url    
    
    # Set up OTLP provider if requested (similar to how Langfuse does it)
    if setup_otlp and api_key:
        try:
            from opentelemetry import trace as otel_trace_api
            
            from ..providers import ZeroEvalOTLPProvider
            
            # Check if there's already a non-proxy tracer provider
            current_provider = otel_trace_api.get_tracer_provider()
            
            if debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true":
                logger = logging.getLogger("zeroeval")
                logger.debug(f"[OTLP SETUP] Current provider type: {type(current_provider).__name__}")
                logger.debug(f"[OTLP SETUP] Is ProxyTracerProvider: {isinstance(current_provider, otel_trace_api.ProxyTracerProvider)}")
            
            if isinstance(current_provider, otel_trace_api.ProxyTracerProvider):
                # Only set up if we have the default/proxy provider
                provider = ZeroEvalOTLPProvider(
                    api_key=api_key,
                    api_url=api_url or os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com"),
                    service_name=service_name
                )
                otel_trace_api.set_tracer_provider(provider)
                
                if debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true":
                    logger = logging.getLogger("zeroeval")
                    logger.debug(f"[OTLP SETUP] ✓ ZeroEvalOTLPProvider configured")
                    logger.debug(f"[OTLP SETUP]   - API URL: {api_url or os.getenv('ZEROEVAL_API_URL', 'https://api.zeroeval.com')}")
                    logger.debug(f"[OTLP SETUP]   - Service name: {service_name}")
                    logger.debug(f"[OTLP SETUP]   - Provider ID: {id(provider)}")
            elif debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true":
                logger = logging.getLogger("zeroeval")
                logger.debug(f"[OTLP SETUP] ⚠️  OTLP provider already configured, skipping setup")
                logger.debug(f"[OTLP SETUP]   - Existing provider: {type(current_provider).__name__}")
                
        except ImportError:
            if debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true":
                logger = logging.getLogger("zeroeval")
                logger.warning("OpenTelemetry not installed, OTLP setup skipped")

    # Ensure a process-level session ID exists for this execution so both SDK and OTEL spans link to it
    try:
        if not os.environ.get("ZEROEVAL_SESSION_ID"):
            os.environ["ZEROEVAL_SESSION_ID"] = str(uuid.uuid4())
        # Respect an existing ZEROEVAL_SESSION_NAME if provided by user; do not auto-generate a name here
    except Exception:
        pass
    
    # Map user-friendly names to actual integration class names
    integration_mapping = {
        "openai": "OpenAIIntegration",
        "gemini": "GeminiIntegration",
        "langchain": "LangChainIntegration", 
        "langgraph": "LangGraphIntegration",
    }
    
    # Handle enabled_integrations - if specified, disable all others
    if enabled_integrations:
        # Get all integration names
        all_integrations = set(integration_mapping.values())
        
        # Convert enabled list to actual class names
        enabled_actual = set()
        for name in enabled_integrations:
            actual_name = integration_mapping.get(name.lower(), name)
            enabled_actual.add(actual_name)
        
        # Disable all integrations NOT in the enabled list
        disabled_actual = all_integrations - enabled_actual
        os.environ["ZEROEVAL_DISABLED_INTEGRATIONS"] = ",".join(disabled_actual)
        
    # Set disabled integrations in environment variable for tracer to pick up
    elif disabled_integrations:
        # Convert user-friendly names to actual class names
        actual_names = []
        for name in disabled_integrations:
            actual_name = integration_mapping.get(name.lower(), name)
            actual_names.append(actual_name)
        
        os.environ["ZEROEVAL_DISABLED_INTEGRATIONS"] = ",".join(actual_names)
    
    # Configure logging
    logger = logging.getLogger("zeroeval")
    
    # Check if debug mode is enabled via param or env var
    is_debug_mode = debug or os.environ.get("ZEROEVAL_DEBUG", "false").lower() == "true"

    if is_debug_mode:
        os.environ["ZEROEVAL_DEBUG"] = "true"  # Ensure env var is set for other modules
        if not logger.handlers: # Avoid adding handlers multiple times
            handler = logging.StreamHandler()
            # Simple, elegant, and readable format with colors
            handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Check which integrations are available
        from ..observability.integrations.gemini.integration import GeminiIntegration
        from ..observability.integrations.httpx.integration import HttpxIntegration
        from ..observability.integrations.langchain.integration import (
            LangChainIntegration,
        )
        from ..observability.integrations.langgraph.integration import (
            LangGraphIntegration,
        )
        from ..observability.integrations.openai.integration import OpenAIIntegration
        
        # List of all integration classes
        integration_classes = [
            OpenAIIntegration,
            GeminiIntegration,
            HttpxIntegration,
            LangChainIntegration,
            LangGraphIntegration,
        ]
        
                # Get the disabled integrations mapping
        integration_mapping = {
            "openai": "OpenAIIntegration",
            "gemini": "GeminiIntegration",
            "httpx": "HttpxIntegration",
            "langchain": "LangChainIntegration",
            "langgraph": "LangGraphIntegration",
        }
        
        # Check which integrations are available and not disabled
        active_integrations = []
        for integration_class in integration_classes:
            integration_name = integration_class.__name__
            # Check if integration is explicitly disabled
            if disabled_integrations:
                # Check if this integration is disabled using either the class name or user-friendly name
                user_friendly_name = next((k for k, v in integration_mapping.items() if v == integration_name), None)
                if (integration_name in [integration_mapping.get(name.lower(), name) for name in disabled_integrations] or
                    (user_friendly_name and user_friendly_name in [name.lower() for name in disabled_integrations])):
                    continue
            
            if integration_class.is_available():
                active_integrations.append(integration_name)
        
        # Log all configuration values as the first log message
        masked_api_key = f"{api_key[:8]}..." if api_key and len(api_key) > 8 else "***" if api_key else "Not set"
        logger.debug("ZeroEval SDK Configuration:")
        logger.debug(f"  Workspace: {workspace_name}")
        logger.debug(f"  API Key: {masked_api_key}")
        logger.debug(f"  API URL: {api_url}")
        logger.debug(f"  Debug Mode: {is_debug_mode}")
        logger.debug(f"  Enabled Integrations: {enabled_integrations or 'All available'}")
        logger.debug(f"  Disabled Integrations: {disabled_integrations or 'None'}")
        logger.debug(f"  Active Integrations: {active_integrations or 'None'}")
        
        logger.info("SDK initialized in debug mode.")
        
        # Apply global tags if provided, then reinitialize integrations
        from ..observability.tracer import tracer
        if tags:
            tracer.set_global_tags(tags)
        tracer.reinitialize_integrations()
    else:
        # If not in debug mode, ensure no logs are shown by default
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING)
        
        # Apply global tags if provided, then reinitialize integrations
        from ..observability.tracer import tracer
        if tags:
            tracer.set_global_tags(tags)
        tracer.reinitialize_integrations()

def _validate_init():
    """Validate the initialization of the ZeroEval SDK."""
    logger = logging.getLogger("zeroeval")
    if not os.environ.get("ZEROEVAL_WORKSPACE_NAME") or not os.environ.get("ZEROEVAL_API_KEY"):
        logger.error(
            "ZeroEval SDK not initialized. Please call ze.init(api_key='YOUR_API_KEY') first."
        )
        return False
    return True

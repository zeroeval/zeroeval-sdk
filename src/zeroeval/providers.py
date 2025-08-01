"""
OpenTelemetry providers for ZeroEval.

This module provides TracerProviders optimized for sending traces to ZeroEval,
with support for different integration scenarios.
"""

import logging
import os
from typing import Any, Optional

from opentelemetry import trace as otel_trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)


class ZeroEvalOTLPProvider(TracerProvider):
    """
    Standard OpenTelemetry TracerProvider configured for ZeroEval.
    
    This provider sets up OTLP export to ZeroEval with proper authentication
    and resource attributes. It works like any standard TracerProvider and
    allows multiple span processors.
    
    Args:
        api_key: ZeroEval API key. If not provided, reads from ZEROEVAL_API_KEY env var.
        api_url: ZeroEval API URL. Defaults to ZEROEVAL_API_URL env var or https://api.zeroeval.com.
        service_name: Service name for traces. Defaults to "zeroeval-app".
        
    Example:
        # Basic usage with environment variables
        provider = ZeroEvalOTLPProvider()
        trace.set_tracer_provider(provider)
        
        # Explicit configuration
        provider = ZeroEvalOTLPProvider(
            api_key="sk_ze_...",
            api_url="https://api.zeroeval.com",
            service_name="my-service"
        )
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_url: Optional[str] = None,
        service_name: str = "zeroeval-app"
    ):
        # Get configuration
        api_key = api_key or os.getenv("ZEROEVAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ZEROEVAL_API_KEY. Set environment variable or pass api_key parameter."
            )
        
        api_url = api_url or os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com")
        endpoint = f"{api_url.rstrip('/')}/v1/traces"
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "0.1.0"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production"),
        })
        
        # Initialize parent TracerProvider
        super().__init__(resource=resource)
        
        # Configure OTLP exporter to ZeroEval
        exporter = OTLPSpanExporter(
            endpoint=endpoint,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        # Add batch processor for efficient span export
        self.add_span_processor(BatchSpanProcessor(exporter))
        
        logger.debug(f"Initialized ZeroEvalOTLPProvider with endpoint: {endpoint}")


class SingleProcessorProvider(ZeroEvalOTLPProvider):
    """
    A ZeroEval provider that only accepts one span processor.
    
    This is useful when integrating with libraries that automatically add their
    own span processors (like Langfuse, Arize, etc.) but you want to ensure
    traces only go to ZeroEval. The first processor added (ZeroEval's) is kept,
    and subsequent processors are silently ignored.
    
    This solves the common "401 Unauthorized" error when using dummy credentials
    with auto-instrumenting libraries.
    
    Example:
        # With Langfuse - prevents duplicate exports
        from langfuse import Langfuse
        
        provider = SingleProcessorProvider()
        langfuse = Langfuse(
            public_key="pk-dummy",
            secret_key="sk-dummy",
            tracer_provider=provider
        )
        
        # Langfuse's auto-added processor is ignored, only ZeroEval receives traces
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processor_locked = True  # Lock after parent adds ZeroEval processor
    
    def add_span_processor(self, span_processor: SpanProcessor) -> None:
        """Add a span processor, but only if we haven't locked yet."""
        if hasattr(self, '_processor_locked') and self._processor_locked:
            processor_name = type(span_processor).__name__
            logger.debug(
                f"Ignoring additional span processor '{processor_name}'. "
                "SingleProcessorProvider only accepts one processor."
            )
            return
            
        super().add_span_processor(span_processor)


def langfuse_zeroeval(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    public_key: str = "-",
    secret_key: str = "-",
    service_name: str = "zeroeval-langfuse",
) -> Any:
    """
    Create a Langfuse client that sends all traces to ZeroEval.
    
    This is the simplest way to use Langfuse with ZeroEval. It handles all the
    configuration automatically and prevents authentication errors.
    
    Args:
        api_key: ZeroEval API key (defaults to ZEROEVAL_API_KEY env var)
        api_url: ZeroEval API URL (defaults to ZEROEVAL_API_URL env var)
        public_key: Placeholder Langfuse public key
        secret_key: Placeholder Langfuse secret key
        service_name: Service name for traces
        
    Returns:
        Configured Langfuse client that sends all traces to ZeroEval
        
    Example:
        from zeroeval.providers import langfuse_zeroeval
        
        # One line setup - reads API key from environment
        langfuse = langfuse_zeroeval()
        
        # Now use Langfuse normally
        from langfuse.openai import openai
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        # All traces automatically go to ZeroEval!
    """
    # Check for API key early for better DX
    api_key = api_key or os.getenv("ZEROEVAL_API_KEY")
    api_url = api_url or os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com")
    if not api_key:
        raise ValueError("langfuse_zeroeval() requires a ZeroEval API key. Set ZEROEVAL_API_KEY environment variable or pass api_key parameter.")
    
    try:
        from langfuse import Langfuse
    except ImportError as e:
        raise ImportError(
            "Langfuse is required. Install with: pip install langfuse"
        ) from e
    
    # Create provider that only accepts one processor (prevents 401 errors)
    provider = SingleProcessorProvider(
        api_key=api_key,
        api_url=api_url,
        service_name=service_name
    )
    
    # Also set globally for any other OTEL instrumentation
    otel_trace_api.set_tracer_provider(provider)
    
    # Return Langfuse client configured with our provider
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        tracer_provider=provider
    )
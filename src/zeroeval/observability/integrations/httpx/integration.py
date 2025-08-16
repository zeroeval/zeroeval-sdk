import json
import logging
import re
import time
from functools import wraps
from typing import Any, Callable, Optional
import contextvars

from ..base import Integration

logger = logging.getLogger(__name__)

# Context variable to track if we're already tracing from the Gemini SDK
try:
    from ..gemini.integration import _gemini_sdk_tracing
except ImportError:
    # If Gemini integration is not available, create a dummy context var
    _gemini_sdk_tracing = contextvars.ContextVar('gemini_sdk_tracing', default=False)


class HttpxIntegration(Integration):
    """
    Integration for httpx to capture network-level API calls to Gemini and other providers.
    
    This integration patches httpx client to automatically create spans for:
    - Gemini REST API calls (generateContent, streamGenerateContent)
    - Other LLM provider API calls (extensible)
    """
    
    PACKAGE_NAME = "httpx"
    
    # Pattern to match Gemini API endpoints
    GEMINI_PATTERN = re.compile(
        r"https://generativelanguage\.googleapis\.com/v\d+(?:beta)?/models/[^/]+:(?:generateContent|streamGenerateContent)"
    )
    
    def setup(self) -> None:
        """Set up httpx integration by patching the client."""
        try:
            import httpx
            
            logger.debug("Setting up HTTPx integration - patching httpx methods")
            
            # Patch sync client
            self._patch_method(
                httpx.Client,
                "request",
                self._wrap_request_sync
            )
            logger.debug("  ✓ Patched httpx.Client.request")
            
            # Patch async client
            self._patch_method(
                httpx.AsyncClient,
                "request",
                self._wrap_request_async
            )
            logger.debug("  ✓ Patched httpx.AsyncClient.request")
            
            # Also patch the module-level functions
            self._patch_method(
                httpx,
                "request",
                self._wrap_request_sync
            )
            logger.debug("  ✓ Patched httpx.request")
            
            # Patch httpx.get, post, etc.
            for method in ["get", "post", "put", "delete", "patch"]:
                if hasattr(httpx, method):
                    self._patch_method(
                        httpx,
                        method,
                        self._wrap_request_sync
                    )
                    logger.debug(f"  ✓ Patched httpx.{method}")
            
            logger.debug("HTTPx integration setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup httpx integration: {e}")
            pass
    
    def _should_trace_request(self, url: str) -> bool:
        """Determine if a request should be traced based on URL."""
        # Check if it's a Gemini API call
        return bool(self.GEMINI_PATTERN.match(url))
    
    def _extract_model_from_url(self, url: str) -> Optional[str]:
        """Extract model name from Gemini API URL."""
        match = re.search(r"/models/([^/:]+):", url)
        if match:
            return match.group(1)
        return None
    
    def _extract_operation_from_url(self, url: str) -> Optional[str]:
        """Extract operation name from Gemini API URL."""
        match = re.search(r":(\w+)$", url)
        if match:
            return match.group(1)
        return None
    
    def _format_contents_as_messages(self, contents: Any) -> list:
        """Format contents as OpenAI-style messages for consistency."""
        if not contents:
            return []
        
        messages = []
        
        # Handle list of content items
        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    # Simple string - assume user message
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    # Handle dict format with role and parts
                    message = {}
                    if "role" in item:
                        message["role"] = item["role"]
                    else:
                        message["role"] = "user"  # Default to user if no role
                    
                    # Extract text content from parts
                    text_parts = []
                    if "parts" in item:
                        for part in item["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                            elif isinstance(part, str):
                                text_parts.append(part)
                    elif "text" in item:
                        text_parts.append(item["text"])
                    
                    message["content"] = "\n".join(text_parts) if text_parts else ""
                    messages.append(message)
        elif isinstance(contents, str):
            # Single string - convert to message
            messages.append({"role": "user", "content": contents})
        
        return messages
    
    def _parse_gemini_request(self, request_data: Any) -> dict[str, Any]:
        """Parse Gemini API request payload."""
        attributes = {}
        
        if isinstance(request_data, (str, bytes)):
            try:
                data = json.loads(request_data)
            except (json.JSONDecodeError, TypeError):
                return attributes
        else:
            data = request_data
        
        # Extract contents
        if "contents" in data:
            attributes["contents"] = data["contents"]
        
        # Extract generation config
        if "generationConfig" in data:
            config = data["generationConfig"]
            if "temperature" in config:
                attributes["temperature"] = config["temperature"]
            if "maxOutputTokens" in config:
                attributes["max_output_tokens"] = config["maxOutputTokens"]
            if "topP" in config:
                attributes["top_p"] = config["topP"]
            if "topK" in config:
                attributes["top_k"] = config["topK"]
            if "stopSequences" in config:
                attributes["stop_sequences"] = config["stopSequences"]
        
        # Extract tools
        if "tools" in data:
            tools_info = []
            for tool in data["tools"]:
                if "functionDeclarations" in tool:
                    for func_decl in tool["functionDeclarations"]:
                        tools_info.append({
                            "name": func_decl.get("name", "unknown"),
                            "description": func_decl.get("description", "")
                        })
            if tools_info:
                attributes["tools"] = tools_info
        
        # Extract tool config
        if "toolConfig" in data:
            tool_config = data["toolConfig"]
            if "functionCallingConfig" in tool_config:
                mode = tool_config["functionCallingConfig"].get("mode")
                if mode:
                    attributes["tool_calling_mode"] = mode
        
        # Extract system instruction
        if "systemInstruction" in data:
            attributes["system_instruction"] = data["systemInstruction"]
        
        # Extract cached content reference
        if "cachedContent" in data:
            attributes["cached_content"] = data["cachedContent"]
        
        return attributes
    
    def _parse_gemini_response(self, response_data: Any) -> dict[str, Any]:
        """Parse Gemini API response payload."""
        attributes = {}
        
        if isinstance(response_data, (str, bytes)):
            try:
                data = json.loads(response_data)
            except (json.JSONDecodeError, TypeError):
                return attributes, None
        else:
            data = response_data
        
        output_text = ""
        function_calls = []
        choices_data = []  # For OpenTelemetry compatibility
        
        # Extract candidates (map to choices for compatibility)
        if "candidates" in data and data["candidates"]:
            for idx, candidate in enumerate(data["candidates"]):
                choice_dict = {"index": idx}
                
                # Extract content
                if "content" in candidate:
                    content = candidate["content"]
                    message_dict = {"role": "assistant", "content": ""}
                    
                    if "parts" in content:
                        for part in content["parts"]:
                            if "text" in part:
                                text = part["text"]
                                message_dict["content"] += text
                                if idx == 0:  # Only accumulate text from first candidate
                                    output_text += text
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                function_calls.append({
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args", {})
                                })
                                # Add to message dict for compatibility
                                if "tool_calls" not in message_dict:
                                    message_dict["tool_calls"] = []
                                message_dict["tool_calls"].append({
                                    "type": "function",
                                    "function": {
                                        "name": fc.get("name", ""),
                                        "arguments": json.dumps(fc.get("args", {})) if isinstance(fc.get("args"), dict) else str(fc.get("args", {}))
                                    }
                                })
                    
                    choice_dict["message"] = message_dict
                
                # Extract finish reason
                if "finishReason" in candidate:
                    choice_dict["finish_reason"] = candidate["finishReason"]
                    attributes["finish_reason"] = candidate["finishReason"]
                
                # Extract safety ratings
                if "safetyRatings" in candidate:
                    safety_ratings = []
                    for rating in candidate["safetyRatings"]:
                        safety_ratings.append({
                            "category": rating.get("category", ""),
                            "probability": rating.get("probability", "")
                        })
                    if idx == 0:  # Only store safety ratings from first candidate in attributes
                        attributes["safety_ratings"] = safety_ratings
                    choice_dict["safety_ratings"] = safety_ratings
                
                choices_data.append(choice_dict)
        
        # Store choices for OpenTelemetry compatibility
        if choices_data:
            attributes["choices"] = choices_data
        
        # Extract usage metadata - map to standard names
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            # Use standard token attribute names
            attributes["input_tokens"] = usage.get("promptTokenCount", 0)
            attributes["output_tokens"] = usage.get("candidatesTokenCount", 0)
            attributes["total_tokens"] = usage.get("totalTokenCount", 0)
            
            # Also store as usage dict for compatibility
            usage_dict = {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
            attributes["usage"] = usage_dict
        
        # Extract model version
        if "modelVersion" in data:
            attributes["model_version"] = data["modelVersion"]
        
        # Extract response ID (Gemini-specific)
        if "responseId" in data:
            attributes["response_id"] = data["responseId"]
            attributes["gemini_response_id"] = data["responseId"]  # Also store with provider prefix
        
        # Extract prompt feedback if present
        if "promptFeedback" in data:
            attributes["prompt_feedback"] = data["promptFeedback"]
        
        if function_calls:
            attributes["function_calls"] = function_calls
        
        # Return both text output and structured data
        # output_text should be used for output_data (UI display)
        # function_calls remain in attributes for structured access
        return attributes, output_text
    
    def _create_tool_spans(self, function_calls: list, tracer: Any) -> None:
        """Create child spans for tool/function calls."""
        for fc in function_calls:
            tool_span = tracer.start_span(
                name=f"tool.{fc['name']}",
                kind="tool",
                attributes={
                    "service.name": "gemini",
                    "type": "tool",
                    "tool_name": fc["name"],
                    "arguments": json.dumps(fc["args"]) if isinstance(fc["args"], dict) else str(fc["args"])
                },
                tags={"integration": "httpx", "tool": fc["name"]}
            )
            tool_span.set_io(
                input_data=json.dumps(fc["args"]) if isinstance(fc["args"], dict) else str(fc["args"]),
                output_data=None
            )
            tracer.end_span(tool_span)
    
    def _wrap_request_sync(self, original: Callable) -> Callable:
        """Wrap synchronous request method."""
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract URL from args/kwargs
            # For Client.request(method, url, ...) - args[0] is self, args[1] is method, args[2] is url
            # For httpx.request(method, url, ...) - args[0] is method, args[1] is url
            url = None
            
            # Check if this is a Client method (first arg is self)
            if args and hasattr(args[0], '__class__') and 'Client' in args[0].__class__.__name__:
                # Client.request(self, method, url, ...)
                if len(args) >= 3:
                    url = str(args[2])  # args[0]=self, args[1]=method, args[2]=url
            elif args:
                # Module-level function: httpx.request(method, url, ...)
                if len(args) >= 2:
                    url = str(args[1])  # args[0]=method, args[1]=url
                elif len(args) == 1 and isinstance(args[0], str):
                    # For module-level functions like httpx.get(url) - but these are handled separately
                    url = str(args[0])
            
            if not url and "url" in kwargs:
                url = str(kwargs["url"])
            
            # Check if we should trace this request
            if not url or not self._should_trace_request(url):
                logger.debug(f"HTTPx request to {url} not traced (doesn't match patterns)")
                return original(*args, **kwargs)
            
            # Skip if we're already tracing from the Gemini SDK
            if _gemini_sdk_tracing.get():
                logger.debug(f"HTTPx request to {url} skipped - already traced by Gemini SDK")
                return original(*args, **kwargs)
            
            logger.debug(f"HTTPx tracing Gemini API request to {url}")
            start_time = time.time()
            
            # Extract model and operation from URL
            model = self._extract_model_from_url(url)
            operation = self._extract_operation_from_url(url)
            is_streaming = operation == "streamGenerateContent"
            
            logger.debug(f"  Model: {model}, Operation: {operation}, Streaming: {is_streaming}")
            
            # Get request body
            request_body = kwargs.get("json") or kwargs.get("data") or kwargs.get("content")
            request_attrs = self._parse_gemini_request(request_body) if request_body else {}
            
            # Start span with LLM-specific attributes
            logger.debug(f"Creating LLM span: gemini.models.{operation or 'generateContent'}")
            
            # Prepare LLM-specific attributes matching backend schema
            span_attributes = {
                # Standard LLM attributes
                "service.name": "gemini",
                "provider": "google",
                "model": model,
                "streaming": is_streaming,
                
                # OpenTelemetry semantic conventions for GenAI
                "gen_ai.system": "gemini",
                "gen_ai.request.model": model,
                "gen_ai.operation.name": "chat",
                
                # HTTP context
                "http.method": "POST",
                "http.url": url,
                
                # Request configuration
                **request_attrs
            }
            
            span = self.tracer.start_span(
                name=f"gemini.models.{operation or 'generateContent'}",
                kind="llm",  # Critical: must be "llm" for proper classification
                attributes=span_attributes,
                tags={"integration": "httpx", "provider": "gemini"},
            )
            
            try:
                response = original(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Handle streaming responses
                if is_streaming:
                    # For streaming, we'll need to wrap the response iterator
                    return _HttpxStreamingResponseProxy(
                        response, span, self.tracer, self, request_body, start_time
                    )
                
                # Parse response for non-streaming
                response_text = response.text
                response_attrs, output = self._parse_gemini_response(response_text)
                
                # Update span attributes
                span.attributes.update(response_attrs)
                
                # Calculate throughput
                if output:
                    throughput = (len(output) / elapsed) if elapsed > 0 else 0
                    span.attributes["throughput"] = round(throughput, 2)
                
                # Set I/O - format like OpenAI integration for consistency
                input_messages = []
                if request_body and isinstance(request_body, dict):
                    if "contents" in request_body:
                        input_messages = self._format_contents_as_messages(request_body["contents"])
                
                # Store input_messages and output_text in attributes for LLMSpanMetrics
                if input_messages:
                    span.attributes["input_messages"] = input_messages
                if output:
                    span.attributes["output_text"] = output
                
                span.set_io(
                    input_data=json.dumps(input_messages) if input_messages else "",  # JSON array of messages like OpenAI
                    output_data=output or ""  # This will be actual text, not function calls JSON
                )
                
                # Create tool spans if there were function calls
                if "function_calls" in response_attrs:
                    self._create_tool_spans(response_attrs["function_calls"], self.tracer)
                
                self.tracer.end_span(span)
                return response
                
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                raise
                
        return wrapper
    
    def _wrap_request_async(self, original: Callable) -> Callable:
        """Wrap asynchronous request method."""
        @wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract URL from args/kwargs
            # For AsyncClient.request(method, url, ...) - args[0] is self, args[1] is method, args[2] is url
            url = None
            
            # Check if this is an AsyncClient method (first arg is self)
            if args and hasattr(args[0], '__class__') and 'Client' in args[0].__class__.__name__:
                # AsyncClient.request(self, method, url, ...)
                if len(args) >= 3:
                    url = str(args[2])  # args[0]=self, args[1]=method, args[2]=url
            elif args and len(args) >= 2:
                # Module-level async function (though less common)
                url = str(args[1])  # args[0]=method, args[1]=url
            
            if not url and "url" in kwargs:
                url = str(kwargs["url"])
            
            # Check if we should trace this request
            if not url or not self._should_trace_request(url):
                return await original(*args, **kwargs)
            
            start_time = time.time()
            
            # Extract model and operation from URL
            model = self._extract_model_from_url(url)
            operation = self._extract_operation_from_url(url)
            is_streaming = operation == "streamGenerateContent"
            
            # Get request body
            request_body = kwargs.get("json") or kwargs.get("data") or kwargs.get("content")
            request_attrs = self._parse_gemini_request(request_body) if request_body else {}
            
            # Start span with LLM-specific attributes
            logger.debug(f"Creating async LLM span: gemini.models.{operation or 'generateContent'}")
            
            # Prepare LLM-specific attributes matching backend schema
            span_attributes = {
                # Standard LLM attributes
                "service.name": "gemini",
                "provider": "google",
                "model": model,
                "streaming": is_streaming,
                
                # OpenTelemetry semantic conventions for GenAI
                "gen_ai.system": "gemini",
                "gen_ai.request.model": model,
                "gen_ai.operation.name": "chat",
                
                # HTTP context
                "http.method": "POST",
                "http.url": url,
                
                # Request configuration
                **request_attrs
            }
            
            span = self.tracer.start_span(
                name=f"gemini.models.{operation or 'generateContent'}",
                kind="llm",  # Critical: must be "llm" for proper classification
                attributes=span_attributes,
                tags={"integration": "httpx", "provider": "gemini"},
            )
            
            try:
                response = await original(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Handle streaming responses
                if is_streaming:
                    # For streaming, we'll need to wrap the response iterator
                    return _HttpxStreamingResponseProxy(
                        response, span, self.tracer, self, request_body, start_time, is_async=True
                    )
                
                # Parse response for non-streaming
                response_text = response.text
                response_attrs, output = self._parse_gemini_response(response_text)
                
                # Update span attributes
                span.attributes.update(response_attrs)
                
                # Calculate throughput
                if output:
                    throughput = (len(output) / elapsed) if elapsed > 0 else 0
                    span.attributes["throughput"] = round(throughput, 2)
                
                # Set I/O - format like OpenAI integration for consistency
                input_messages = []
                if request_body and isinstance(request_body, dict):
                    if "contents" in request_body:
                        input_messages = self._format_contents_as_messages(request_body["contents"])
                
                # Store input_messages and output_text in attributes for LLMSpanMetrics
                if input_messages:
                    span.attributes["input_messages"] = input_messages
                if output:
                    span.attributes["output_text"] = output
                
                span.set_io(
                    input_data=json.dumps(input_messages) if input_messages else "",  # JSON array of messages like OpenAI
                    output_data=output or ""  # This will be actual text, not function calls JSON
                )
                
                # Create tool spans if there were function calls
                if "function_calls" in response_attrs:
                    self._create_tool_spans(response_attrs["function_calls"], self.tracer)
                
                self.tracer.end_span(span)
                return response
                
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                raise
                
        return wrapper


class _HttpxStreamingResponseProxy:
    """Proxy for httpx streaming responses to capture SSE data."""
    
    def __init__(self, response, span, tracer, integration, request_body, start_time, is_async=False):
        self._response = response
        self._span = span
        self._tracer = tracer
        self._integration = integration
        self._request_body = request_body
        self._start_time = start_time
        self._is_async = is_async
        self._accumulated_text = ""
        self._function_calls = []
        self._usage_metadata = None
        self._finished = False
    
    def __getattr__(self, name):
        """Proxy all other attributes to the original response."""
        return getattr(self._response, name)
    
    def iter_lines(self):
        """Iterate over SSE lines for sync streaming."""
        for line in self._response.iter_lines():
            # Process SSE data
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str and data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        self._process_chunk(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
            yield line
        
        self._finish_span()
    
    async def aiter_lines(self):
        """Iterate over SSE lines for async streaming."""
        async for line in self._response.aiter_lines():
            # Process SSE data
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str and data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        self._process_chunk(data)
                    except (json.JSONDecodeError, TypeError):
                        pass
            yield line
        
        self._finish_span()
    
    def _process_chunk(self, data: dict[str, Any]):
        """Process a streaming chunk."""
        # Extract candidates
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            
            # Extract content
            if "content" in candidate:
                content = candidate["content"]
                if "parts" in content:
                    for part in content["parts"]:
                        if "text" in part:
                            self._accumulated_text += part["text"]
                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            self._function_calls.append({
                                "name": fc.get("name", ""),
                                "args": fc.get("args", {})
                            })
        
        # Extract usage metadata
        if "usageMetadata" in data:
            self._usage_metadata = data["usageMetadata"]
    
    def _finish_span(self):
        """Finish the span with accumulated data."""
        if self._finished:
            return
        self._finished = True
        
        elapsed = time.time() - self._start_time
        
        # Update span with usage data
        if self._usage_metadata:
            # Use standard field names for token counts
            self._span.attributes["input_tokens"] = self._usage_metadata.get("promptTokenCount", 0)
            self._span.attributes["output_tokens"] = self._usage_metadata.get("candidatesTokenCount", 0)
            self._span.attributes["total_tokens"] = self._usage_metadata.get("totalTokenCount", 0)
            
            # Also store as usage dict for compatibility
            usage_dict = {
                "prompt_tokens": self._usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": self._usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": self._usage_metadata.get("totalTokenCount", 0)
            }
            self._span.attributes["usage"] = usage_dict
        
        # Set throughput
        throughput = (len(self._accumulated_text) / elapsed) if (self._accumulated_text and elapsed > 0) else 0
        self._span.attributes["throughput"] = round(throughput, 2)
        
        # Set function calls
        if self._function_calls:
            self._span.attributes["function_calls"] = self._function_calls
            # Create tool spans
            self._integration._create_tool_spans(self._function_calls, self._tracer)
        
        # Set I/O data - format like OpenAI integration for consistency
        input_messages = []
        if self._request_body and isinstance(self._request_body, dict):
            if "contents" in self._request_body:
                input_messages = self._integration._format_contents_as_messages(self._request_body["contents"])
        
        # Store input_messages and output_text in attributes for LLMSpanMetrics
        if input_messages:
            self._span.attributes["input_messages"] = input_messages
        if self._accumulated_text:
            self._span.attributes["output_text"] = self._accumulated_text
        
        self._span.set_io(
            input_data=json.dumps(input_messages) if input_messages else "",  # JSON array of messages like OpenAI
            output_data=self._accumulated_text  # Use actual text, not function calls JSON
        )
        
        self._tracer.end_span(self._span)

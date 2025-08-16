import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, Dict, List
import time
import contextvars

from ..base import Integration

logger = logging.getLogger(__name__)

# Context variable to track if we're already tracing from the SDK
_gemini_sdk_tracing = contextvars.ContextVar('gemini_sdk_tracing', default=False)


class GeminiIntegration(Integration):
    """
    Integration for Google's Generative AI (Gemini) Python client library.
    
    Patches Gemini client to automatically create spans for:
    - Content generation (sync and async)
    - Streaming responses
    - Tool/function calls
    - Structured outputs
    - Token usage and safety ratings
    """
    
    PACKAGE_NAME = "google.genai"

    def setup(self) -> None:
        """Set up Gemini integration by patching the client."""
        try:
            import google.genai
            
            # Patch Client initialization
            self._patch_method(
                google.genai.Client,
                "__init__",
                self._wrap_init
            )
        except Exception as e:
            logger.error(f"Failed to setup Gemini integration: {e}")
            pass

    def _wrap_init(self, original: Callable) -> Callable:
        """Wrap Gemini client initialization to patch generation methods."""
        @wraps(original)
        def wrapper(client_instance, *args: Any, **kwargs: Any) -> Any:
            # Call original init
            result = original(client_instance, *args, **kwargs)
            
            # Patch sync generate_content
            self._patch_method(
                client_instance.models,
                "generate_content",
                self._wrap_generate_content_sync
            )
            
            # Patch sync streaming generate_content_stream  
            if hasattr(client_instance.models, 'generate_content_stream'):
                self._patch_method(
                    client_instance.models,
                    "generate_content_stream",
                    self._wrap_generate_content_stream_sync
                )
            
            # Patch async methods if available
            if hasattr(client_instance, 'aio') and hasattr(client_instance.aio, 'models'):
                self._patch_method(
                    client_instance.aio.models,
                    "generate_content",
                    self._wrap_generate_content_async
                )
                
                if hasattr(client_instance.aio.models, 'generate_content_stream'):
                    self._patch_method(
                        client_instance.aio.models,
                        "generate_content_stream", 
                        self._wrap_generate_content_stream_async
                    )
            
            return result
        return wrapper

    def _serialize_contents(self, contents: Any) -> Any:
        """Serialize contents for storage."""
        if contents is None:
            return None
            
        # Handle string content
        if isinstance(contents, str):
            return contents
            
        # Handle list of contents
        if isinstance(contents, list):
            serialized = []
            for item in contents:
                if isinstance(item, str):
                    serialized.append(item)
                elif hasattr(item, '__dict__'):
                    # Try to convert objects to dict
                    try:
                        serialized.append(self._object_to_dict(item))
                    except:
                        serialized.append(str(item))
                else:
                    serialized.append(item)
            return serialized
            
        # Handle single Content object
        if hasattr(contents, '__dict__'):
            try:
                return self._object_to_dict(contents)
            except:
                return str(contents)
                
        return contents
    
    def _format_contents_as_messages(self, contents: Any) -> Any:
        """Format contents as OpenAI-style messages for consistency."""
        if contents is None:
            return []
            
        # Handle string content - convert to message format
        if isinstance(contents, str):
            return [{"role": "user", "content": contents}]
            
        # Handle list of contents
        if isinstance(contents, list):
            messages = []
            for item in contents:
                if isinstance(item, str):
                    # Simple string - assume user message
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    # Handle dict format with role and parts
                    message = {}
                    if 'role' in item:
                        message['role'] = item['role']
                    else:
                        message['role'] = 'user'  # Default to user if no role
                    
                    # Extract text content from parts
                    text_parts = []
                    if 'parts' in item:
                        for part in item['parts']:
                            if isinstance(part, dict) and 'text' in part:
                                text_parts.append(part['text'])
                            elif isinstance(part, str):
                                text_parts.append(part)
                    elif 'text' in item:
                        text_parts.append(item['text'])
                    
                    message['content'] = '\n'.join(text_parts) if text_parts else ''
                    messages.append(message)
                elif hasattr(item, 'role') and hasattr(item, 'parts'):
                    # Handle Content object
                    message = {'role': getattr(item, 'role', 'user')}
                    text_parts = []
                    for part in item.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                        elif isinstance(part, str):
                            text_parts.append(part)
                    message['content'] = '\n'.join(text_parts) if text_parts else ''
                    messages.append(message)
            return messages
            
        # Handle single Content object
        if hasattr(contents, 'parts'):
            text_parts = []
            for part in contents.parts:
                if hasattr(part, 'text'):
                    text_parts.append(part.text)
                elif isinstance(part, str):
                    text_parts.append(part)
            return [{"role": "user", "content": '\n'.join(text_parts) if text_parts else ''}]
                
        # Default: convert to string and wrap in message
        return [{"role": "user", "content": str(contents) if contents else ''}]

    def _object_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert Gemini objects to dictionaries."""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    if hasattr(value, '__dict__'):
                        result[key] = self._object_to_dict(value)
                    elif isinstance(value, list):
                        result[key] = [self._object_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                    else:
                        result[key] = value
            return result
        return obj

    def _extract_config_attributes(self, config: Any) -> Dict[str, Any]:
        """Extract attributes from GenerateContentConfig."""
        attributes = {}
        
        if not config:
            return attributes
            
        # Extract common configuration
        if hasattr(config, 'temperature'):
            attributes['temperature'] = getattr(config, 'temperature', None)
        if hasattr(config, 'max_output_tokens'):
            attributes['max_output_tokens'] = getattr(config, 'max_output_tokens', None)
        if hasattr(config, 'top_p'):
            attributes['top_p'] = getattr(config, 'top_p', None)
        if hasattr(config, 'top_k'):
            attributes['top_k'] = getattr(config, 'top_k', None)
        if hasattr(config, 'stop_sequences'):
            attributes['stop_sequences'] = getattr(config, 'stop_sequences', None)
            
        # Extract response schema if present
        if hasattr(config, 'response_schema'):
            schema = getattr(config, 'response_schema', None)
            if schema:
                attributes['response_schema'] = str(schema)
        if hasattr(config, 'response_mime_type'):
            attributes['response_mime_type'] = getattr(config, 'response_mime_type', None)
            
        # Extract tools if present
        if hasattr(config, 'tools'):
            tools = getattr(config, 'tools', None)
            if tools:
                tools_info = []
                for tool in tools:
                    if hasattr(tool, 'function_declarations') and tool.function_declarations:
                        for func_decl in tool.function_declarations:
                            tools_info.append({
                                'name': getattr(func_decl, 'name', 'unknown'),
                                'description': getattr(func_decl, 'description', '')
                            })
                    elif callable(tool):
                        # Python function
                        tools_info.append({
                            'name': tool.__name__,
                            'description': tool.__doc__ or ''
                        })
                attributes['tools'] = tools_info
                
        # Extract tool config if present
        if hasattr(config, 'tool_config'):
            tool_config = getattr(config, 'tool_config', None)
            if tool_config and hasattr(tool_config, 'function_calling_config'):
                fcc = tool_config.function_calling_config
                if hasattr(fcc, 'mode'):
                    attributes['tool_calling_mode'] = getattr(fcc, 'mode', None)
                    
        return attributes

    def _wrap_generate_content_sync(self, original: Callable) -> Callable:
        """Wrap synchronous generate_content method."""
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            # Set a context flag to prevent duplicate tracing by httpx integration
            token = _gemini_sdk_tracing.set(True)
            
            # Extract arguments
            model = kwargs.get('model', args[0] if args else None)
            contents = kwargs.get('contents', args[1] if len(args) > 1 else None)
            config = kwargs.get('config', args[2] if len(args) > 2 else None)
            
            # Extract config attributes
            config_attrs = self._extract_config_attributes(config)
            
            # Start span
            span = self.tracer.start_span(
                name="gemini.models.generate_content",
                kind="llm",
                attributes={
                    "service.name": "gemini",
                    "provider": "google",
                    "model": model,
                    "contents": self._serialize_contents(contents),
                    "streaming": False,
                    **config_attrs
                },
                tags={"integration": "gemini"},
            )
            
            try:
                response = original(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Extract response metadata
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Extract content
                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        output_text = ""
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    output_text += part.text
                        
                        # Check for function calls and create child spans
                        function_calls = []
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                if hasattr(part, 'function_call'):
                                    fc = part.function_call
                                    function_call_data = {
                                        'name': getattr(fc, 'name', ''),
                                        'args': getattr(fc, 'args', {})
                                    }
                                    function_calls.append(function_call_data)
                                    
                                    # Create child span for tool call
                                    tool_span = self.tracer.start_span(
                                        name=f"tool.{function_call_data['name']}",
                                        kind="tool",
                                        attributes={
                                            "service.name": "gemini",
                                            "type": "tool",
                                            "tool_name": function_call_data['name'],
                                            "arguments": json.dumps(function_call_data['args']) if isinstance(function_call_data['args'], dict) else str(function_call_data['args'])
                                        },
                                        tags={"integration": "gemini", "tool": function_call_data['name']}
                                    )
                                    tool_span.set_io(
                                        input_data=json.dumps(function_call_data['args']) if isinstance(function_call_data['args'], dict) else str(function_call_data['args']),
                                        output_data=None  # Will be filled by actual tool execution
                                    )
                                    self.tracer.end_span(tool_span)
                        
                        if function_calls:
                            span.attributes["function_calls"] = function_calls
                            
                        # Set throughput
                        throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                        span.attributes["throughput"] = round(throughput, 2)
                        
                        # Set IO - format like OpenAI integration for consistency
                        input_messages = self._format_contents_as_messages(contents)
                        
                        # Store input_messages and output_text in attributes for LLMSpanMetrics
                        if input_messages:
                            span.attributes["input_messages"] = input_messages
                        if output_text:
                            span.attributes["output_text"] = output_text
                        
                        span.set_io(
                            input_data=json.dumps(input_messages),  # JSON array of messages like OpenAI
                            output_data=output_text  # Use actual text, not function calls JSON
                        )
                    
                    # Extract finish reason
                    if hasattr(candidate, 'finish_reason'):
                        span.attributes["finish_reason"] = str(candidate.finish_reason)
                    
                    # Extract safety ratings
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        safety_ratings = []
                        for rating in candidate.safety_ratings:
                            safety_ratings.append({
                                'category': str(getattr(rating, 'category', '')),
                                'probability': str(getattr(rating, 'probability', ''))
                            })
                        span.attributes["safety_ratings"] = safety_ratings
                
                # Extract usage metadata - use standard field names
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    span.attributes["input_tokens"] = getattr(usage, 'prompt_token_count', 0)
                    span.attributes["output_tokens"] = getattr(usage, 'candidates_token_count', 0)
                    span.attributes["total_tokens"] = getattr(usage, 'total_token_count', 0)
                    
                    usage_dict = {
                        'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                        'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                        'total_tokens': getattr(usage, 'total_token_count', 0)
                    }
                    span.attributes["usage"] = usage_dict
                
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                return response
                
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                raise
                
        return wrapper

    def _wrap_generate_content_stream_sync(self, original: Callable) -> Callable:
        """Wrap synchronous streaming generate_content_stream method."""
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            # Set a context flag to prevent duplicate tracing by httpx integration
            token = _gemini_sdk_tracing.set(True)
            
            # Extract arguments
            model = kwargs.get('model', args[0] if args else None)
            contents = kwargs.get('contents', args[1] if len(args) > 1 else None)
            config = kwargs.get('config', args[2] if len(args) > 2 else None)
            
            # Extract config attributes
            config_attrs = self._extract_config_attributes(config)
            
            # Start span
            span = self.tracer.start_span(
                name="gemini.models.generate_content_stream",
                kind="llm",
                attributes={
                    "service.name": "gemini",
                    "provider": "google",
                    "model": model,
                    "contents": self._serialize_contents(contents),
                    "streaming": True,
                    **config_attrs
                },
                tags={"integration": "gemini"},
            )
            
            try:
                response = original(*args, **kwargs)
                # Return a streaming proxy to capture the response
                # Pass the token to reset it when streaming completes
                return _StreamingResponseProxy(response, span, self.tracer, self, contents, start_time, is_async=False, context_token=token)
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                raise
                
        return wrapper

    def _wrap_generate_content_async(self, original: Callable) -> Callable:
        """Wrap asynchronous generate_content method."""
        @wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            # Set a context flag to prevent duplicate tracing by httpx integration
            token = _gemini_sdk_tracing.set(True)
            
            # Extract arguments
            model = kwargs.get('model', args[0] if args else None)
            contents = kwargs.get('contents', args[1] if len(args) > 1 else None)
            config = kwargs.get('config', args[2] if len(args) > 2 else None)
            
            # Extract config attributes
            config_attrs = self._extract_config_attributes(config)
            
            # Start span
            span = self.tracer.start_span(
                name="gemini.models.generate_content",
                kind="llm",
                attributes={
                    "service.name": "gemini",
                    "provider": "google",
                    "model": model,
                    "contents": self._serialize_contents(contents),
                    "streaming": False,
                    **config_attrs
                },
                tags={"integration": "gemini"},
            )
            
            try:
                response = await original(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Extract response metadata (same as sync version)
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Extract content
                    if hasattr(candidate, 'content'):
                        content = candidate.content
                        output_text = ""
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    output_text += part.text
                        
                        # Check for function calls and create child spans
                        function_calls = []
                        if hasattr(content, 'parts') and content.parts:
                            for part in content.parts:
                                if hasattr(part, 'function_call'):
                                    fc = part.function_call
                                    function_call_data = {
                                        'name': getattr(fc, 'name', ''),
                                        'args': getattr(fc, 'args', {})
                                    }
                                    function_calls.append(function_call_data)
                                    
                                    # Create child span for tool call
                                    tool_span = self.tracer.start_span(
                                        name=f"tool.{function_call_data['name']}",
                                        kind="tool",
                                        attributes={
                                            "service.name": "gemini",
                                            "type": "tool",
                                            "tool_name": function_call_data['name'],
                                            "arguments": json.dumps(function_call_data['args']) if isinstance(function_call_data['args'], dict) else str(function_call_data['args'])
                                        },
                                        tags={"integration": "gemini", "tool": function_call_data['name']}
                                    )
                                    tool_span.set_io(
                                        input_data=json.dumps(function_call_data['args']) if isinstance(function_call_data['args'], dict) else str(function_call_data['args']),
                                        output_data=None
                                    )
                                    self.tracer.end_span(tool_span)
                        
                        if function_calls:
                            span.attributes["function_calls"] = function_calls
                            
                        # Set throughput
                        throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                        span.attributes["throughput"] = round(throughput, 2)
                        
                        # Set IO - format like OpenAI integration for consistency
                        input_messages = self._format_contents_as_messages(contents)
                        
                        # Store input_messages and output_text in attributes for LLMSpanMetrics
                        if input_messages:
                            span.attributes["input_messages"] = input_messages
                        if output_text:
                            span.attributes["output_text"] = output_text
                        
                        span.set_io(
                            input_data=json.dumps(input_messages),  # JSON array of messages like OpenAI
                            output_data=output_text  # Use actual text, not function calls JSON
                        )
                    
                    # Extract finish reason
                    if hasattr(candidate, 'finish_reason'):
                        span.attributes["finish_reason"] = str(candidate.finish_reason)
                    
                    # Extract safety ratings
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        safety_ratings = []
                        for rating in candidate.safety_ratings:
                            safety_ratings.append({
                                'category': str(getattr(rating, 'category', '')),
                                'probability': str(getattr(rating, 'probability', ''))
                            })
                        span.attributes["safety_ratings"] = safety_ratings
                
                # Extract usage metadata - use standard field names
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    span.attributes["input_tokens"] = getattr(usage, 'prompt_token_count', 0)
                    span.attributes["output_tokens"] = getattr(usage, 'candidates_token_count', 0)
                    span.attributes["total_tokens"] = getattr(usage, 'total_token_count', 0)
                    
                    usage_dict = {
                        'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                        'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                        'total_tokens': getattr(usage, 'total_token_count', 0)
                    }
                    span.attributes["usage"] = usage_dict
                
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                return response
                
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                raise
                
        return wrapper

    def _wrap_generate_content_stream_async(self, original: Callable) -> Callable:
        """Wrap asynchronous streaming generate_content_stream method."""
        @wraps(original)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            # Set a context flag to prevent duplicate tracing by httpx integration
            token = _gemini_sdk_tracing.set(True)
            
            # Extract arguments
            model = kwargs.get('model', args[0] if args else None)
            contents = kwargs.get('contents', args[1] if len(args) > 1 else None)
            config = kwargs.get('config', args[2] if len(args) > 2 else None)
            
            # Extract config attributes
            config_attrs = self._extract_config_attributes(config)
            
            # Start span
            span = self.tracer.start_span(
                name="gemini.models.generate_content_stream",
                kind="llm",
                attributes={
                    "service.name": "gemini",
                    "provider": "google",
                    "model": model,
                    "contents": self._serialize_contents(contents),
                    "streaming": True,
                    **config_attrs
                },
                tags={"integration": "gemini"},
            )
            
            try:
                response = await original(*args, **kwargs)
                # Return a streaming proxy to capture the response
                # Pass the token to reset it when streaming completes
                return _StreamingResponseProxy(response, span, self.tracer, self, contents, start_time, is_async=True, context_token=token)
            except Exception as e:
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None)
                )
                self.tracer.end_span(span)
                _gemini_sdk_tracing.reset(token)  # Reset the context flag
                raise
                
        return wrapper


# Streaming response proxy for Gemini
class _StreamingResponseProxy:
    """Proxy for Gemini streaming responses to capture usage and create spans."""
    
    def __init__(self, response, span, tracer, integration, contents, start_time, is_async=False, context_token=None):
        self._response = response
        self._span = span
        self._tracer = tracer
        self._integration = integration
        self._contents = contents
        self._start_time = start_time
        self._is_async = is_async
        self._context_token = context_token
        self._chunks = []
        self._usage_metadata = None
        self._finished = False
        self._function_calls = []
        
    async def __aiter__(self):
        """Async iteration for streaming responses."""
        accumulated_text = ""
        
        async for chunk in self._response:
            self._chunks.append(chunk)
            
            # Extract text from chunk
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text'):
                                accumulated_text += part.text
                            elif hasattr(part, 'function_call'):
                                fc = part.function_call
                                self._function_calls.append({
                                    'name': getattr(fc, 'name', ''),
                                    'args': getattr(fc, 'args', {})
                                })
            
            # Extract usage metadata if available
            if hasattr(chunk, 'usage_metadata'):
                self._usage_metadata = chunk.usage_metadata
            
            yield chunk
        
        # Create tool call spans after streaming completes
        for function_call in self._function_calls:
            tool_span = self._tracer.start_span(
                name=f"tool.{function_call['name']}",
                kind="tool",
                attributes={
                    "service.name": "gemini",
                    "type": "tool",
                    "tool_name": function_call['name'],
                    "arguments": json.dumps(function_call['args']) if isinstance(function_call['args'], dict) else str(function_call['args'])
                },
                tags={"integration": "gemini", "tool": function_call['name']}
            )
            tool_span.set_io(
                input_data=json.dumps(function_call['args']) if isinstance(function_call['args'], dict) else str(function_call['args']),
                output_data=None
            )
            self._tracer.end_span(tool_span)
        
        self._finish_span(accumulated_text)
        
    def __iter__(self):
        """Sync iteration for streaming responses."""
        accumulated_text = ""
        
        for chunk in self._response:
            self._chunks.append(chunk)
            
            # Extract text from chunk
            if hasattr(chunk, 'candidates') and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    if hasattr(content, 'parts') and content.parts:
                        for part in content.parts:
                            if hasattr(part, 'text'):
                                accumulated_text += part.text
                            elif hasattr(part, 'function_call'):
                                fc = part.function_call
                                self._function_calls.append({
                                    'name': getattr(fc, 'name', ''),
                                    'args': getattr(fc, 'args', {})
                                })
            
            # Extract usage metadata if available
            if hasattr(chunk, 'usage_metadata'):
                self._usage_metadata = chunk.usage_metadata
            
            yield chunk
        
        # Create tool call spans after streaming completes
        for function_call in self._function_calls:
            tool_span = self._tracer.start_span(
                name=f"tool.{function_call['name']}",
                kind="tool",
                attributes={
                    "service.name": "gemini",
                    "type": "tool",
                    "tool_name": function_call['name'],
                    "arguments": json.dumps(function_call['args']) if isinstance(function_call['args'], dict) else str(function_call['args'])
                },
                tags={"integration": "gemini", "tool": function_call['name']}
            )
            tool_span.set_io(
                input_data=json.dumps(function_call['args']) if isinstance(function_call['args'], dict) else str(function_call['args']),
                output_data=None
            )
            self._tracer.end_span(tool_span)
        
        self._finish_span(accumulated_text)
        
    def _finish_span(self, accumulated_text: str):
        """Finish the span with accumulated data."""
        if self._finished:
            return
        self._finished = True
        
        elapsed = time.time() - self._start_time
        
        # Update span with usage data - use standard field names
        if self._usage_metadata:
            self._span.attributes["input_tokens"] = getattr(self._usage_metadata, 'prompt_token_count', 0)
            self._span.attributes["output_tokens"] = getattr(self._usage_metadata, 'candidates_token_count', 0)
            self._span.attributes["total_tokens"] = getattr(self._usage_metadata, 'total_token_count', 0)
            
            usage_dict = {
                'prompt_tokens': getattr(self._usage_metadata, 'prompt_token_count', 0),
                'completion_tokens': getattr(self._usage_metadata, 'candidates_token_count', 0),
                'total_tokens': getattr(self._usage_metadata, 'total_token_count', 0)
            }
            self._span.attributes["usage"] = usage_dict
        
        # Set throughput
        throughput = (len(accumulated_text) / elapsed) if (accumulated_text and elapsed > 0) else 0
        self._span.attributes["throughput"] = round(throughput, 2)
        
        # Set function calls
        if self._function_calls:
            self._span.attributes["function_calls"] = self._function_calls
        
        # Set I/O data - format like OpenAI integration for consistency
        input_messages = self._integration._format_contents_as_messages(self._contents)
        
        # Store input_messages and output_text in attributes for LLMSpanMetrics
        if input_messages:
            self._span.attributes["input_messages"] = input_messages
        if accumulated_text:
            self._span.attributes["output_text"] = accumulated_text
        
        self._span.set_io(
            input_data=json.dumps(input_messages),  # JSON array of messages like OpenAI
            output_data=accumulated_text  # Use actual text, not function calls JSON
        )
        
        self._tracer.end_span(self._span)
        
        # Reset the context flag if we have a token
        if self._context_token is not None:
            _gemini_sdk_tracing.reset(self._context_token)
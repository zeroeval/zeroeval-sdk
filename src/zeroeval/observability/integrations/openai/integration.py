import logging
from functools import wraps
from typing import Any, Callable, Optional

from ..base import Integration

logger = logging.getLogger(__name__)


class OpenAIIntegration(Integration):
    """
    Integration for OpenAI's Python client library.
    
    Patches OpenAI client to automatically create spans for:
    - Chat completions (sync and async)
    - Streaming responses  
    - Tool/function calls
    - Token usage and costs
    """
    
    PACKAGE_NAME = "openai"

    def setup(self) -> None:
        """Set up OpenAI integration by patching the client."""
        try:
            import openai
            
            # Patch both sync and async clients
            self._patch_method(
                openai.OpenAI,
                "__init__",
                self._wrap_init
            )
            self._patch_method(
                openai.AsyncOpenAI,
                "__init__",
                self._wrap_init
            )
        except Exception as e:
            logger.error(f"Failed to setup OpenAI integration: {e}")
            pass

    def _wrap_init(self, original: Callable) -> Callable:
        """Wrap OpenAI client initialization to patch completion methods."""
        @wraps(original)
        def wrapper(client_instance, *args: Any, **kwargs: Any) -> Any:
            # Call original init
            result = original(client_instance, *args, **kwargs)
            
            # Patch chat.completions.create for this specific instance
            self._patch_method(
                client_instance.chat.completions,
                "create", 
                self._wrap_chat_completion_sync(client_instance.chat.completions)
            )
            
            # The create method handles both sync and async internally
            
            return result
        return wrapper

    def _serialize_messages(self, messages: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Serialize messages for storage, handling tool calls and other special content."""
        if not messages:
            return []
        
        serialized = []
        for msg in messages:
            serialized_msg = {
                "role": msg.get("role"),
                "content": msg.get("content")
            }
            
            # Include tool calls if present
            if "tool_calls" in msg:
                serialized_msg["tool_calls"] = msg["tool_calls"]
            
            # Include tool call id if present
            if "tool_call_id" in msg:
                serialized_msg["tool_call_id"] = msg["tool_call_id"]
                
            # Include name if present (for function/tool messages)
            if "name" in msg:
                serialized_msg["name"] = msg["name"]
                
            serialized.append(serialized_msg)
            
        return serialized

    def _extract_tool_calls(self, message) -> Optional[list[dict[str, Any]]]:
        """Extract tool calls from a message object."""
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return None
            
        tool_calls = []
        for tool_call in message.tool_calls:
            tool_calls.append({
                'id': getattr(tool_call, 'id', None),
                'type': getattr(tool_call, 'type', 'function'),
                'function': {
                    'name': getattr(tool_call.function, 'name', None),
                    'arguments': getattr(tool_call.function, 'arguments', None)
                } if hasattr(tool_call, 'function') else None
            })
        return tool_calls

    # ------------------------------------------------------------------+
    #  Async wrapper – client = openai.AsyncOpenAI                      |
    # ------------------------------------------------------------------+
    def _wrap_chat_completion_async(self, original: Callable) -> Callable:  # noqa: C901 (length)
        import json
        import time

        @wraps(original)
        async def wrapper(*args: Any, **kwargs: Any):
            start_time = time.time()
            is_streaming = kwargs.get("stream", False)

            if is_streaming and (not isinstance(kwargs.get("model"), str) or "/" not in kwargs["model"]):
                kwargs["stream_options"] = {"include_usage": True}

            # Try to get base_url from client instance
            base_url = None
            if args and hasattr(args[0], 'base_url'):
                base_url = str(args[0].base_url)

            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                kind="llm",
                attributes={
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": self._serialize_messages(kwargs.get("messages")),
                    "streaming": is_streaming,
                    "base_url": base_url,
                },
                tags={"integration": "openai"},
            )
            tracer = self.tracer

            try:
                response = await original(*args, **kwargs)
                if is_streaming:
                    logger.debug("Async wrapper -> returning _StreamingResponseProxy (async)")
                    return _StreamingResponseProxy(
                        response, span, tracer, self, kwargs, start_time, is_async=True
                    )

                # ---------- non-streaming ----------
                elapsed = time.time() - start_time
                usage = getattr(response, "usage", None)
                if usage:
                    span.attributes.update(
                        {"inputTokens": usage.prompt_tokens, "outputTokens": usage.completion_tokens}
                    )
                
                # Capture additional OpenAI response data
                if hasattr(response, 'id'):
                    span.attributes["openai_id"] = response.id
                if hasattr(response, 'system_fingerprint'):
                    span.attributes["system_fingerprint"] = response.system_fingerprint
                if hasattr(response, 'choices') and response.choices:
                    # Convert choices to dict for JSON serialization
                    choices_data = []
                    for choice in response.choices:
                        choice_dict = {}
                        if hasattr(choice, 'message'):
                            message_dict = {
                                'role': getattr(choice.message, 'role', None),
                                'content': getattr(choice.message, 'content', None)
                            }
                            
                            # Capture tool calls if present
                            tool_calls = self._extract_tool_calls(choice.message)
                            if tool_calls:
                                message_dict['tool_calls'] = tool_calls
                                
                                # Create child spans for each tool call
                                for tool_call in tool_calls:
                                    if tool_call.get('function'):
                                        tool_span = tracer.start_span(
                                            name=f"tool.{tool_call['function']['name']}",
                                            kind="tool",
                                            attributes={
                                                "service.name": "openai",
                                                "type": "tool",
                                                "tool_name": tool_call['function']['name'],
                                                "tool_call_id": tool_call.get('id'),
                                                "arguments": tool_call['function'].get('arguments')
                                            },
                                            tags={"integration": "openai", "tool": tool_call['function']['name']}
                                        )
                                        tool_span.set_io(
                                            input_data=tool_call['function'].get('arguments'),
                                            output_data=None  # Will be filled by the actual tool execution
                                        )
                                        tracer.end_span(tool_span)
                            
                            choice_dict['message'] = message_dict
                        if hasattr(choice, 'finish_reason'):
                            choice_dict['finish_reason'] = choice.finish_reason
                        if hasattr(choice, 'index'):
                            choice_dict['index'] = choice.index
                        choices_data.append(choice_dict)
                    span.attributes["choices"] = choices_data
                if usage:
                    # Store full usage object
                    usage_dict = {
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': getattr(usage, 'total_tokens', usage.prompt_tokens + usage.completion_tokens)
                    }
                    span.attributes["usage"] = usage_dict
                
                message = response.choices[0].message if response.choices else None
                output = message.content if message else None
                
                # If no content but has tool calls, format tool calls as output
                if not output and hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls_output = []
                    for tc in message.tool_calls:
                        tool_calls_output.append({
                            'tool': tc.function.name if hasattr(tc, 'function') else 'unknown',
                            'arguments': tc.function.arguments if hasattr(tc, 'function') else '{}'
                        })
                    output = json.dumps(tool_calls_output, indent=2)
                
                throughput = (len(output) / elapsed) if (output and elapsed > 0) else 0
                span.attributes["throughput"] = round(throughput, 2)
                span.set_io(
                    input_data=json.dumps(self._serialize_messages(kwargs.get("messages"))),
                    output_data=output,
                )
                tracer.end_span(span)
                return response
            except Exception as e:
                span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                tracer.end_span(span)
                raise

        return wrapper

    # ------------------------------------------------------------------+
    #  Sync wrapper – client = openai.OpenAI                            |
    # ------------------------------------------------------------------+
    def _wrap_chat_completion_sync(self, completions_instance) -> Callable:  # noqa: C901 (length)
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any):
                start_time = time.time()
                is_streaming = kwargs.get("stream", False)

                if is_streaming and (not isinstance(kwargs.get("model"), str) or "/" not in kwargs["model"]):
                    kwargs["stream_options"] = {"include_usage": True}

                # Try to get base_url from client instance
                base_url = None
                if hasattr(completions_instance, '_client') and hasattr(completions_instance._client, 'base_url'):
                    base_url = str(completions_instance._client.base_url)

                span = self.tracer.start_span(
                name="openai.chat.completions.create",
                kind="llm",
                attributes={
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": self._serialize_messages(kwargs.get("messages")),
                    "streaming": is_streaming,
                    "base_url": base_url,
                },
                tags={"integration": "openai"},
            )
                tracer = self.tracer

                try:
                    response = original(*args, **kwargs)
                    if is_streaming:
                        logger.debug("Sync wrapper -> returning _StreamingResponseProxy (sync)")
                        return _StreamingResponseProxy(
                            response, span, tracer, self, kwargs, start_time, is_async=False
                        )

                    # ---------- non-streaming ----------
                    elapsed = time.time() - start_time
                    usage = getattr(response, "usage", None)
                    if usage:
                        span.attributes.update(
                            {"inputTokens": usage.prompt_tokens, "outputTokens": usage.completion_tokens}
                        )
                    
                    # Capture additional OpenAI response data
                    if hasattr(response, 'id'):
                        span.attributes["openai_id"] = response.id
                    if hasattr(response, 'system_fingerprint'):
                        span.attributes["system_fingerprint"] = response.system_fingerprint
                    if hasattr(response, 'choices') and response.choices:
                        # Convert choices to dict for JSON serialization
                        choices_data = []
                        for choice in response.choices:
                            choice_dict = {}
                            if hasattr(choice, 'message'):
                                message_dict = {
                                    'role': getattr(choice.message, 'role', None),
                                    'content': getattr(choice.message, 'content', None)
                                }
                                
                                # Capture tool calls if present
                                tool_calls = self._extract_tool_calls(choice.message)
                                if tool_calls:
                                    message_dict['tool_calls'] = tool_calls
                                    
                                    # Create child spans for each tool call
                                    for tool_call in tool_calls:
                                        if tool_call.get('function'):
                                            tool_span = tracer.start_span(
                                                name=f"tool.{tool_call['function']['name']}",
                                                kind="tool",
                                                attributes={
                                                    "service.name": "openai",
                                                    "type": "tool",
                                                    "tool_name": tool_call['function']['name'],
                                                    "tool_call_id": tool_call.get('id'),
                                                    "arguments": tool_call['function'].get('arguments')
                                                },
                                                tags={"integration": "openai", "tool": tool_call['function']['name']}
                                            )
                                            tool_span.set_io(
                                                input_data=tool_call['function'].get('arguments'),
                                                output_data=None  # Will be filled by the actual tool execution
                                            )
                                            tracer.end_span(tool_span)
                                
                                choice_dict['message'] = message_dict
                            if hasattr(choice, 'finish_reason'):
                                choice_dict['finish_reason'] = choice.finish_reason
                            if hasattr(choice, 'index'):
                                choice_dict['index'] = choice.index
                            choices_data.append(choice_dict)
                        span.attributes["choices"] = choices_data
                    if usage:
                        # Store full usage object
                        usage_dict = {
                            'prompt_tokens': usage.prompt_tokens,
                            'completion_tokens': usage.completion_tokens,
                            'total_tokens': getattr(usage, 'total_tokens', usage.prompt_tokens + usage.completion_tokens)
                        }
                        span.attributes["usage"] = usage_dict
                    
                    message = response.choices[0].message if response.choices else None
                    output = message.content if message else None
                    
                    # If no content but has tool calls, format tool calls as output
                    if not output and hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_output = []
                        for tc in message.tool_calls:
                            tool_calls_output.append({
                                'tool': tc.function.name if hasattr(tc, 'function') else 'unknown',
                                'arguments': tc.function.arguments if hasattr(tc, 'function') else '{}'
                            })
                        output = json.dumps(tool_calls_output, indent=2)
                    
                    throughput = (len(output) / elapsed) if (output and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(kwargs.get("messages"))),
                        output_data=output,
                    )
                    tracer.end_span(span)
                    return response
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise
            return inner
        return wrapper


# Streaming response proxy for OpenAI
class _StreamingResponseProxy:
    """Proxy for OpenAI streaming responses to capture usage and create spans."""
    
    def __init__(self, response, span, tracer, integration, kwargs, start_time, is_async=False):
        self._response = response
        self._span = span
        self._tracer = tracer
        self._integration = integration
        self._kwargs = kwargs
        self._start_time = start_time
        self._is_async = is_async
        self._chunks = []
        self._usage = None
        self._finished = False
        self._accumulated_content = ""
        self._tool_calls_by_index = {}

    async def __aiter__(self):
        """Async iteration for streaming responses."""
        accumulated_content = ""
        tool_calls_by_index = self._tool_calls_by_index
        
        async for chunk in self._response:
            self._chunks.append(chunk)
            
            # Accumulate content
            if chunk.choices:
                for choice in chunk.choices:
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                        accumulated_content += choice.delta.content
                    
                    # Handle streaming tool calls
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tool_call_delta in choice.delta.tool_calls:
                            index = getattr(tool_call_delta, 'index', 0)
                            if index not in tool_calls_by_index:
                                tool_calls_by_index[index] = {
                                    'id': getattr(tool_call_delta, 'id', None),
                                    'type': getattr(tool_call_delta, 'type', 'function'),
                                    'function': {
                                        'name': '',
                                        'arguments': ''
                                    }
                                }
                            
                            if hasattr(tool_call_delta, 'function'):
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    tool_calls_by_index[index]['function']['name'] = tool_call_delta.function.name
                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    tool_calls_by_index[index]['function']['arguments'] += tool_call_delta.function.arguments
            
            # Extract usage if available
            if hasattr(chunk, 'usage') and chunk.usage:
                self._usage = chunk.usage
            
            yield chunk
        
        # Create tool call spans after streaming completes
        if tool_calls_by_index:
            for tool_call in tool_calls_by_index.values():
                if tool_call.get('function') and tool_call['function'].get('name'):
                    tool_span = self._tracer.start_span(
                        name=f"tool.{tool_call['function']['name']}",
                        kind="tool",
                        attributes={
                            "service.name": "openai",
                            "type": "tool",
                            "tool_name": tool_call['function']['name'],
                            "tool_call_id": tool_call.get('id'),
                            "arguments": tool_call['function'].get('arguments')
                        },
                        tags={"integration": "openai", "tool": tool_call['function']['name']}
                    )
                    tool_span.set_io(
                        input_data=tool_call['function'].get('arguments'),
                        output_data=None
                    )
                    self._tracer.end_span(tool_span)
        
        self._finish_span(accumulated_content)

    def __iter__(self):
        """Sync iteration for streaming responses."""
        accumulated_content = ""
        tool_calls_by_index = self._tool_calls_by_index
        
        for chunk in self._response:
            self._chunks.append(chunk)
            
            # Accumulate content
            if chunk.choices:
                for choice in chunk.choices:
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                        accumulated_content += choice.delta.content
                    
                    # Handle streaming tool calls
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        for tool_call_delta in choice.delta.tool_calls:
                            index = getattr(tool_call_delta, 'index', 0)
                            if index not in tool_calls_by_index:
                                tool_calls_by_index[index] = {
                                    'id': getattr(tool_call_delta, 'id', None),
                                    'type': getattr(tool_call_delta, 'type', 'function'),
                                    'function': {
                                        'name': '',
                                        'arguments': ''
                                    }
                                }
                            
                            if hasattr(tool_call_delta, 'function'):
                                if hasattr(tool_call_delta.function, 'name') and tool_call_delta.function.name:
                                    tool_calls_by_index[index]['function']['name'] = tool_call_delta.function.name
                                if hasattr(tool_call_delta.function, 'arguments') and tool_call_delta.function.arguments:
                                    tool_calls_by_index[index]['function']['arguments'] += tool_call_delta.function.arguments
            
            # Extract usage if available
            if hasattr(chunk, 'usage') and chunk.usage:
                self._usage = chunk.usage
            
            yield chunk
        
        # Create tool call spans after streaming completes
        if tool_calls_by_index:
            for tool_call in tool_calls_by_index.values():
                if tool_call.get('function') and tool_call['function'].get('name'):
                    tool_span = self._tracer.start_span(
                        name=f"tool.{tool_call['function']['name']}",
                        kind="tool",
                        attributes={
                            "service.name": "openai",
                            "type": "tool",
                            "tool_name": tool_call['function']['name'],
                            "tool_call_id": tool_call.get('id'),
                            "arguments": tool_call['function'].get('arguments')
                        },
                        tags={"integration": "openai", "tool": tool_call['function']['name']}
                    )
                    tool_span.set_io(
                        input_data=tool_call['function'].get('arguments'),
                        output_data=None
                    )
                    self._tracer.end_span(tool_span)
        
        self._finish_span(accumulated_content)
    
    def _finish_span(self, accumulated_content: str):
        """Finish the span with accumulated data."""
        if self._finished:
            return
        self._finished = True
        
        import json
        import time
        
        elapsed = time.time() - self._start_time
        
        # Update span with usage data
        if self._usage:
            self._span.attributes.update({
                "inputTokens": self._usage.prompt_tokens,
                "outputTokens": self._usage.completion_tokens,
            })
            usage_dict = {
                'prompt_tokens': self._usage.prompt_tokens,
                'completion_tokens': self._usage.completion_tokens,
                'total_tokens': getattr(self._usage, 'total_tokens', self._usage.prompt_tokens + self._usage.completion_tokens)
            }
            self._span.attributes["usage"] = usage_dict
        
        # Set throughput
        throughput = (len(accumulated_content) / elapsed) if (accumulated_content and elapsed > 0) else 0
        self._span.attributes["throughput"] = round(throughput, 2)
        
        # Set I/O data - if no content but has tool calls, show tool calls as output
        output = accumulated_content
        if not output and self._tool_calls_by_index:
            tool_calls_output = []
            for tc in self._tool_calls_by_index.values():
                if tc.get('function') and tc['function'].get('name'):
                    tool_calls_output.append({
                        'tool': tc['function']['name'],
                        'arguments': tc['function'].get('arguments', '{}')
                    })
            output = json.dumps(tool_calls_output, indent=2)
        
        self._span.set_io(
            input_data=json.dumps(self._integration._serialize_messages(self._kwargs.get("messages"))),
            output_data=output
        )
        
        self._tracer.end_span(self._span)
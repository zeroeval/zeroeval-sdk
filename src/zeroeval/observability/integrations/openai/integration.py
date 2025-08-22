import json
import logging
import re
from functools import wraps
from typing import Any, Callable, Optional

from ..base import Integration

logger = logging.getLogger(__name__)


def zeroeval_prompt(prompt: str, variables: Optional[dict] = None, task: Optional[str] = None) -> str:
    """
    Helper function to create a prompt with zeroeval metadata.
    
    Args:
        prompt: The actual prompt content (e.g., "You are a helpful assistant.")
        variables: Dictionary of variables to be interpolated in the prompt
        task: Optional task identifier for this prompt
    
    Returns:
        A string with the format: <zeroeval>{JSON}</zeroeval>prompt
        
    Example:
        >>> zeroeval_prompt(
        ...     "You are a helpful assistant. The price is {{price}}",
        ...     variables={"price": 10},
        ...     task="pricing_assistant"
        ... )
        '<zeroeval>{"variables": {"price": 10}, "task": "pricing_assistant"}</zeroeval>You are a helpful assistant. The price is {{price}}'
    """
    metadata = {}
    
    if variables:
        metadata["variables"] = variables
    
    if task:
        metadata["task"] = task
    
    if metadata:
        return f'<zeroeval>{json.dumps(metadata)}</zeroeval>{prompt}'
    
    return prompt


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
            # Choose async vs sync wrapper based on the client type
            try:
                import openai as _openai  # local import to avoid hard dependency at module load
                is_async_client = isinstance(client_instance, _openai.AsyncOpenAI)
            except Exception:
                # Fallback: detect by attribute commonly present on async client
                is_async_client = hasattr(client_instance, "_client") and hasattr(client_instance._client, "_asynchronous")

            if is_async_client:
                # Use async-aware wrapper factory (returns an async function)
                self._patch_method(
                    client_instance.chat.completions,
                    "create",
                    self._wrap_chat_completion_async()
                )
            else:
                # Use sync wrapper factory
                self._patch_method(
                    client_instance.chat.completions,
                    "create", 
                    self._wrap_chat_completion_sync(client_instance.chat.completions)
                )
            
            # The create method is now correctly wrapped for the client's sync/async nature
            
            # Also patch responses.create if available (for GPT-5 and newer models)
            if hasattr(client_instance, 'responses'):
                logger.debug("[OpenAI] Found responses endpoint, patching responses.create")
                if is_async_client:
                    self._patch_method(
                        client_instance.responses,
                        "create",
                        self._wrap_responses_create_async()
                    )
                else:
                    self._patch_method(
                        client_instance.responses,
                        "create",
                        self._wrap_responses_create_sync(client_instance.responses)
                    )
            
            return result
        return wrapper

    def _extract_zeroeval_metadata(self, content: str) -> tuple[Optional[dict[str, Any]], str]:
        """
        Extract <zeroeval> metadata from content and return (metadata, cleaned_content).
        
        Returns:
            - Tuple of (metadata dict or None, cleaned content string)
        """
        # Look for <zeroeval>...</zeroeval> tags
        pattern = r'<zeroeval>(.*?)</zeroeval>'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None, content
        
        try:
            # Extract and parse the JSON
            json_str = match.group(1).strip()
            metadata = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
            
            # Remove the <zeroeval> tags from content
            cleaned_content = re.sub(pattern, '', content, count=1).strip()
            
            return metadata, cleaned_content
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse zeroeval metadata: {e}")
            raise ValueError(f"Invalid JSON in <zeroeval> tags: {e}") from e
    
    def _interpolate_variables(self, text: str, variables: dict[str, Any]) -> str:
        """
        Interpolate variables in text using {{variable_name}} syntax.
        
        Args:
            text: Text containing {{variable_name}} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Text with variables interpolated
        """
        if not variables or not text:
            return text
        
        # Replace {{variable_name}} with the actual value
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            text = text.replace(placeholder, str(value))
        
        return text
    
    def _process_messages_with_zeroeval(self, messages: Optional[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """
        Process messages to extract zeroeval metadata and interpolate variables.
        
        Returns:
            - Tuple of (processed messages, zeroeval metadata)
        """
        if not messages:
            return messages, None
        
        # Deep copy messages to avoid modifying the original
        import copy
        processed_messages = copy.deepcopy(messages)
        zeroeval_metadata = None
        variables = {}
        
        # Check if first message is a system message with zeroeval tags
        if processed_messages and processed_messages[0].get("role") == "system":
            content = processed_messages[0].get("content", "")
            
            # Extract zeroeval metadata
            metadata, cleaned_content = self._extract_zeroeval_metadata(content)
            
            if metadata:
                zeroeval_metadata = metadata
                variables = metadata.get("variables", {})
                
                # Update the first message with cleaned content
                processed_messages[0]["content"] = cleaned_content
                
                # Log extraction
                logger.debug(f"Extracted zeroeval metadata: task={metadata.get('task')}, variables={variables}")
        
        # Interpolate variables in all messages if we have any
        if variables:
            for msg in processed_messages:
                if msg.get("content"):
                    msg["content"] = self._interpolate_variables(msg["content"], variables)
        
        return processed_messages, zeroeval_metadata

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

    def _serialize_responses_input(self, input_data: Any) -> list[dict[str, Any]]:
        """Serialize Responses API input into a chat-like messages structure.

        The Responses API accepts a flexible `input` parameter. We normalize it to
        a list of messages shaped like OpenAI Chat messages: {role, content}.
        """
        import json as _json

        if input_data is None:
            return []

        # Simple string -> single user message
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]

        # List of items
        if isinstance(input_data, list):
            messages: list[dict[str, Any]] = []
            for item in input_data:
                # Typical message item: { role, content: [ {type, text, ...}, ... ] }
                if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                    role = item.get("role")
                    content = item.get("content")
                    # content may be a list of content parts; extract text-like parts
                    if isinstance(content, list):
                        parts: list[str] = []
                        for part in content:
                            if isinstance(part, dict):
                                # Prefer text fields (input_text.text, text)
                                if "text" in part and isinstance(part["text"], str):
                                    parts.append(part["text"]) 
                                else:
                                    # Fallback: include a compact JSON for non-text parts
                                    try:
                                        parts.append(_json.dumps(part))
                                    except Exception:
                                        parts.append(str(part))
                            else:
                                parts.append(str(part))
                        messages.append({"role": role, "content": "\n".join(parts)})
                    else:
                        # If content is a plain string
                        messages.append({"role": role, "content": str(content)})
                    continue

                # Direct text item (e.g., {type: "input_text", text: "..."})
                if isinstance(item, dict) and item.get("type") in ("input_text", "text") and item.get("text"):
                    messages.append({"role": "user", "content": item.get("text")})
                    continue

                # Unknown shape -> include compact JSON under user role to preserve data
                try:
                    messages.append({"role": "user", "content": _json.dumps(item)})
                except Exception:
                    messages.append({"role": "user", "content": str(item)})

            return messages

        # Dict or other types -> best-effort stringify
        try:
            return [{"role": "user", "content": _json.dumps(input_data)}]
        except Exception:
            return [{"role": "user", "content": str(input_data)}]

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
    def _wrap_chat_completion_async(self) -> Callable:  # noqa: C901 (length)
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            async def inner(*args: Any, **kwargs: Any):
                start_time = time.time()
                is_streaming = kwargs.get("stream", False)
                
                # Store original messages for span attributes
                original_messages = kwargs.get("messages")
                
                # Process messages to extract zeroeval metadata and interpolate variables
                processed_messages, zeroeval_metadata = self._process_messages_with_zeroeval(original_messages)
                if processed_messages is not None:
                    kwargs["messages"] = processed_messages

                # Always add stream_options for OpenAI streaming calls to get token usage
                if is_streaming:
                    # Only add if not already present and not a custom/fine-tuned model
                    model = kwargs.get("model", "")
                    if "stream_options" not in kwargs and (isinstance(model, str) and "/" not in model):
                        kwargs["stream_options"] = {"include_usage": True}
                        logger.debug(f"[OpenAI] Added stream_options for model: {model}")

                # Try to get base_url from client instance
                base_url = None
                if args and hasattr(args[0], 'base_url'):
                    base_url = str(args[0].base_url)

                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "streaming": is_streaming,
                    "base_url": base_url,
                }
                
                # Always capture tools and tool_choice if present in the request
                if "tools" in kwargs and kwargs["tools"]:
                    # Store tool definitions for rendering
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if tool.get("type") == "function" and "function" in tool:
                            tools_info.append({
                                "type": "function",
                                "name": tool["function"].get("name"),
                                "description": tool["function"].get("description")
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]  # Keep raw format too
                
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                
                # Add zeroeval metadata to attributes if present
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    span_attributes["task"] = zeroeval_metadata.get("task")
                    # Store the original system prompt template (with {{variables}})
                    if original_messages and original_messages[0].get("role") == "system":
                        # Extract just the content after the zeroeval tags
                        _, template_content = self._extract_zeroeval_metadata(original_messages[0].get("content", ""))
                        if template_content:
                            span_attributes["system_prompt_template"] = template_content
                    
                span = self.tracer.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    attributes=span_attributes,
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
                                    # Store tool calls in main span attributes
                                    if 'tool_calls' not in span.attributes:
                                        span.attributes['tool_calls'] = []
                                    span.attributes['tool_calls'].extend(tool_calls)
                                    
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
                        # Store token counts for cost calculation
                        span.attributes["inputTokens"] = usage.prompt_tokens
                        span.attributes["outputTokens"] = usage.completion_tokens
                        
                        # Store full usage object
                        usage_dict = {
                            'prompt_tokens': usage.prompt_tokens,
                            'completion_tokens': usage.completion_tokens,
                            'total_tokens': getattr(usage, 'total_tokens', usage.prompt_tokens + usage.completion_tokens)
                        }
                        span.attributes["usage"] = usage_dict
                    
                    message = response.choices[0].message if response.choices else None
                    output = message.content if message else None
                    
                    # If there are tool calls, include them in a structured format
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_data = self._extract_tool_calls(message)
                        if tool_calls_data:
                            # If no content, show tool calls as primary output
                            if not output:
                                output = json.dumps({
                                    "tool_calls": tool_calls_data
                                }, indent=2)
                            else:
                                # If there's content AND tool calls, show both
                                output = json.dumps({
                                    "content": output,
                                    "tool_calls": tool_calls_data
                                }, indent=2)
                    
                    throughput = (len(output) / elapsed) if (output and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(kwargs.get("messages"))),
                        output_data=output,
                    )
                    tracer.end_span(span)
                    
                    # Check if response needs wrapping (for OpenAI-compatible APIs that return dicts)
                    if isinstance(response, dict) and not hasattr(response, 'to_dict'):
                        # Wrap dictionary responses to provide OpenAI-compatible methods
                        return _ResponseWrapper(response)
                    
                    return response
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise

            return inner

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
                
                # Store original messages for span attributes
                original_messages = kwargs.get("messages")
                
                # Process messages to extract zeroeval metadata and interpolate variables
                processed_messages, zeroeval_metadata = self._process_messages_with_zeroeval(original_messages)
                if processed_messages is not None:
                    kwargs["messages"] = processed_messages

                # Always add stream_options for OpenAI streaming calls to get token usage
                if is_streaming:
                    # Only add if not already present and not a custom/fine-tuned model
                    model = kwargs.get("model", "")
                    if "stream_options" not in kwargs and (isinstance(model, str) and "/" not in model):
                        kwargs["stream_options"] = {"include_usage": True}
                        logger.debug(f"[OpenAI] Added stream_options for model: {model}")

                # Try to get base_url from client instance
                base_url = None
                if hasattr(completions_instance, '_client') and hasattr(completions_instance._client, 'base_url'):
                    base_url = str(completions_instance._client.base_url)

                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "streaming": is_streaming,
                    "base_url": base_url,
                }
                
                # Always capture tools and tool_choice if present in the request
                if "tools" in kwargs and kwargs["tools"]:
                    # Store tool definitions for rendering
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if tool.get("type") == "function" and "function" in tool:
                            tools_info.append({
                                "type": "function",
                                "name": tool["function"].get("name"),
                                "description": tool["function"].get("description")
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]  # Keep raw format too
                
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                
                # Add zeroeval metadata to attributes if present
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    span_attributes["task"] = zeroeval_metadata.get("task")
                    # Store the original system prompt template (with {{variables}})
                    if original_messages and original_messages[0].get("role") == "system":
                        # Extract just the content after the zeroeval tags
                        _, template_content = self._extract_zeroeval_metadata(original_messages[0].get("content", ""))
                        if template_content:
                            span_attributes["system_prompt_template"] = template_content
                    
                span = self.tracer.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    attributes=span_attributes,
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
                                    # Store tool calls in main span attributes
                                    if 'tool_calls' not in span.attributes:
                                        span.attributes['tool_calls'] = []
                                    span.attributes['tool_calls'].extend(tool_calls)
                                    
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
                        # Store token counts for cost calculation
                        span.attributes["inputTokens"] = usage.prompt_tokens
                        span.attributes["outputTokens"] = usage.completion_tokens
                        
                        # Store full usage object
                        usage_dict = {
                            'prompt_tokens': usage.prompt_tokens,
                            'completion_tokens': usage.completion_tokens,
                            'total_tokens': getattr(usage, 'total_tokens', usage.prompt_tokens + usage.completion_tokens)
                        }
                        span.attributes["usage"] = usage_dict
                    
                    message = response.choices[0].message if response.choices else None
                    output = message.content if message else None
                    
                    # If there are tool calls, include them in a structured format
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        tool_calls_data = self._extract_tool_calls(message)
                        if tool_calls_data:
                            # If no content, show tool calls as primary output
                            if not output:
                                output = json.dumps({
                                    "tool_calls": tool_calls_data
                                }, indent=2)
                            else:
                                # If there's content AND tool calls, show both
                                output = json.dumps({
                                    "content": output,
                                    "tool_calls": tool_calls_data
                                }, indent=2)
                    
                    throughput = (len(output) / elapsed) if (output and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(kwargs.get("messages"))),
                        output_data=output,
                    )
                    tracer.end_span(span)
                    
                    # Check if response needs wrapping (for OpenAI-compatible APIs that return dicts)
                    if isinstance(response, dict) and not hasattr(response, 'to_dict'):
                        # Wrap dictionary responses to provide OpenAI-compatible methods
                        return _ResponseWrapper(response)
                    
                    return response
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise
            return inner
        return wrapper
    
    # ------------------------------------------------------------------+
    #  Responses API wrappers (for GPT-5 and newer models)              |
    # ------------------------------------------------------------------+
    def _wrap_responses_create_async(self) -> Callable:
        """Wrap responses.create for async clients (GPT-5 models)."""
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            async def inner(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                
                # Extract input and normalize to messages
                raw_input = kwargs.get("input")
                normalized_messages = self._serialize_responses_input(raw_input)
                
                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "endpoint": "responses",
                }
                # Try to capture base_url if available (async client passes resource instance in args[0])
                base_url = None
                if args and hasattr(args[0], "_client") and hasattr(args[0]._client, "base_url"):
                    try:
                        base_url = str(args[0]._client.base_url)
                    except Exception:
                        base_url = None
                if base_url:
                    span_attributes["base_url"] = base_url
                
                # Capture tools if present
                if "tools" in kwargs and kwargs["tools"]:
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if isinstance(tool, dict) and tool.get("type") == "function":
                            tools_info.append({
                                "type": "function",
                                "name": tool.get("function", {}).get("name"),
                                "description": tool.get("function", {}).get("description")
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                # Store normalized messages for UI convenience
                if normalized_messages:
                    span_attributes["messages"] = normalized_messages
                
                # Capture reasoning config if present
                if "reasoning" in kwargs:
                    span_attributes["reasoning"] = kwargs["reasoning"]
                
                span = self.tracer.start_span(
                    name="openai.responses.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses"},
                )
                tracer = self.tracer

                try:
                    response = await original(*args, **kwargs)
                    
                    elapsed = time.time() - start_time
                    
                    # Extract usage data
                    usage = getattr(response, "usage", None)
                    if usage:
                        # Note: responses API uses input_tokens/output_tokens
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                        span.attributes.update({
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                        })
                        
                        usage_dict = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': getattr(usage, 'total_tokens', input_tokens + output_tokens)
                        }
                        span.attributes["usage"] = usage_dict
                    
                    # Extract output text
                    output_text = getattr(response, "output_text", "")
                    # Capture basic response identifiers when available
                    if hasattr(response, "id"):
                        span.attributes["openai_id"] = response.id
                    if hasattr(response, "system_fingerprint"):
                        span.attributes["system_fingerprint"] = response.system_fingerprint
                    
                    # Extract tool calls from output
                    output_items = getattr(response, "output", [])
                    tool_calls = []
                    reasoning_trace = []
                    
                    for item in output_items:
                        if hasattr(item, "type"):
                            if item.type == "function_call":
                                # Align with chat.completions tool_calls shape
                                tool_call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                                tool_call = {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": getattr(item, "name", None),
                                        "arguments": getattr(item, "arguments", "")
                                    }
                                }
                                tool_calls.append(tool_call)
                            elif item.type == "reasoning" and hasattr(item, "summary"):
                                for summary in item.summary:
                                    if hasattr(summary, "text"):
                                        reasoning_trace.append(summary.text)
                    
                    if tool_calls:
                        span.attributes["tool_calls"] = tool_calls
                        # Also reflect tool calls in a synthetic assistant message so UI can render them
                        try:
                            existing_messages = span.attributes.get("messages") or normalized_messages or []
                        except Exception:
                            existing_messages = normalized_messages or []
                        try:
                            messages_with_assistant = list(existing_messages)
                        except Exception:
                            messages_with_assistant = []
                        messages_with_assistant.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls
                        })
                        span.attributes["messages"] = messages_with_assistant
                        # Create child spans for each tool call (consistent with chat completions)
                        for tc in tool_calls:
                            fn = tc.get("function") or {}
                            tool_span = tracer.start_span(
                                name=f"tool.{fn.get('name')}",
                                kind="tool",
                                attributes={
                                    "service.name": "openai",
                                    "type": "tool",
                                    "tool_name": fn.get("name"),
                                    "tool_call_id": tc.get("id"),
                                    "arguments": fn.get("arguments"),
                                },
                                tags={"integration": "openai", "tool": fn.get("name") or "function"},
                            )
                            tool_span.set_io(input_data=fn.get("arguments"), output_data=None)
                            tracer.end_span(tool_span)
                        
                    if reasoning_trace:
                        span.attributes["reasoning_trace"] = reasoning_trace
                    
                    # Set throughput
                    throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)
                    
                    # Set I/O data
                    # To avoid duplicate UI bubbles, do not stringify tool_calls into output_data.
                    # If tool calls exist, keep output_data as plain content (or None if no content).
                    combined_output = output_text if output_text else (None if tool_calls else None)

                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(normalized_messages)),
                        output_data=combined_output,
                    )
                    
                    tracer.end_span(span)
                    
                    # Check if response needs wrapping
                    if isinstance(response, dict) and not hasattr(response, 'to_dict'):
                        return _ResponseWrapper(response)
                    
                    return response
                    
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise

            return inner
        return wrapper
    
    def _wrap_responses_create_sync(self, responses_instance) -> Callable:
        """Wrap responses.create for sync clients (GPT-5 models)."""
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                
                # Extract input messages/prompts and normalize
                raw_input = kwargs.get("input")
                normalized_messages = self._serialize_responses_input(raw_input)
                
                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "endpoint": "responses",
                }
                # base_url from client instance if available
                base_url = None
                if hasattr(responses_instance, "_client") and hasattr(responses_instance._client, "base_url"):
                    try:
                        base_url = str(responses_instance._client.base_url)
                    except Exception:
                        base_url = None
                if base_url:
                    span_attributes["base_url"] = base_url
                
                # Capture tools if present
                if "tools" in kwargs and kwargs["tools"]:
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if isinstance(tool, dict) and tool.get("type") == "function":
                            tools_info.append({
                                "type": "function",
                                "name": tool.get("function", {}).get("name"),
                                "description": tool.get("function", {}).get("description")
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                # Store normalized messages for UI convenience
                if normalized_messages:
                    span_attributes["messages"] = normalized_messages
                
                # Capture reasoning config if present
                if "reasoning" in kwargs:
                    span_attributes["reasoning"] = kwargs["reasoning"]
                
                span = self.tracer.start_span(
                    name="openai.responses.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses"},
                )
                tracer = self.tracer

                try:
                    response = original(*args, **kwargs)
                    
                    elapsed = time.time() - start_time
                    
                    # Extract usage data
                    usage = getattr(response, "usage", None)
                    if usage:
                        # Note: responses API uses input_tokens/output_tokens
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                        span.attributes.update({
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                        })
                        
                        usage_dict = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': getattr(usage, 'total_tokens', input_tokens + output_tokens)
                        }
                        span.attributes["usage"] = usage_dict
                    
                    # Extract output text
                    output_text = getattr(response, "output_text", "")
                    # Capture basic response identifiers when available
                    if hasattr(response, "id"):
                        span.attributes["openai_id"] = response.id
                    if hasattr(response, "system_fingerprint"):
                        span.attributes["system_fingerprint"] = response.system_fingerprint
                    
                    # Extract tool calls from output
                    output_items = getattr(response, "output", [])
                    tool_calls = []
                    reasoning_trace = []
                    
                    for item in output_items:
                        if hasattr(item, "type"):
                            if item.type == "function_call":
                                # Align with chat.completions tool_calls shape
                                tool_call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                                tool_call = {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": getattr(item, "name", None),
                                        "arguments": getattr(item, "arguments", "")
                                    }
                                }
                                tool_calls.append(tool_call)
                            elif item.type == "reasoning" and hasattr(item, "summary"):
                                for summary in item.summary:
                                    if hasattr(summary, "text"):
                                        reasoning_trace.append(summary.text)
                    
                    if tool_calls:
                        span.attributes["tool_calls"] = tool_calls
                        # Also reflect tool calls in a synthetic assistant message so UI can render them
                        try:
                            existing_messages = span.attributes.get("messages") or normalized_messages or []
                        except Exception:
                            existing_messages = normalized_messages or []
                        try:
                            messages_with_assistant = list(existing_messages)
                        except Exception:
                            messages_with_assistant = []
                        messages_with_assistant.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls
                        })
                        span.attributes["messages"] = messages_with_assistant
                        # Create child spans for each tool call (consistent with chat completions)
                        for tc in tool_calls:
                            fn = tc.get("function") or {}
                            tool_span = tracer.start_span(
                                name=f"tool.{fn.get('name')}",
                                kind="tool",
                                attributes={
                                    "service.name": "openai",
                                    "type": "tool",
                                    "tool_name": fn.get("name"),
                                    "tool_call_id": tc.get("id"),
                                    "arguments": fn.get("arguments"),
                                },
                                tags={"integration": "openai", "tool": fn.get("name") or "function"},
                            )
                            tool_span.set_io(input_data=fn.get("arguments"), output_data=None)
                            tracer.end_span(tool_span)
                        
                    if reasoning_trace:
                        span.attributes["reasoning_trace"] = reasoning_trace
                    
                    # Set throughput
                    throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)
                    
                    # Set I/O data
                    # To avoid duplicate UI bubbles, do not stringify tool_calls into output_data.
                    # If tool calls exist, keep output_data as plain content (or None if no content).
                    combined_output = output_text if output_text else (None if tool_calls else None)

                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(normalized_messages)),
                        output_data=combined_output,
                    )
                    
                    tracer.end_span(span)
                    
                    # Check if response needs wrapping
                    if isinstance(response, dict) and not hasattr(response, 'to_dict'):
                        return _ResponseWrapper(response)
                    
                    return response
                    
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise

            return inner
        return wrapper


# Streaming response proxy for OpenAI
class _ResponseWrapper:
    """Base wrapper class that provides OpenAI-compatible response methods."""
    
    def __init__(self, response_data: dict):
        self._response_data = response_data
        # Store all attributes from the response data
        for key, value in response_data.items():
            setattr(self, key, value)
    
    def to_dict(
        self,
        *,
        mode: str = "python",
        use_api_names: bool = True,
        exclude_unset: bool = True,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a dictionary representation of the response.
        Compatible with OpenAI's to_dict() method.
        
        Args:
            mode: "json" for JSON-serializable types, "python" for any Python objects
            use_api_names: Whether to use API field names
            exclude_unset: Whether to exclude fields that were not set
            exclude_defaults: Whether to exclude fields with default values
            exclude_none: Whether to exclude None values
            warnings: Whether to show warnings
        
        Returns:
            Dictionary representation of the response
        """
        result = {}
        for key, value in self._response_data.items():
            if exclude_none and value is None:
                continue
            if mode == "json" and hasattr(value, "isoformat"):
                # Convert datetime to string for JSON mode
                value = value.isoformat()
            result[key] = value
        return result
    
    def to_json(
        self,
        *,
        indent: Optional[int] = 2,
        use_api_names: bool = True,
        exclude_unset: bool = True,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        warnings: bool = True,
    ) -> str:
        """
        Generate a JSON string representation of the response.
        Compatible with OpenAI's to_json() method.
        
        Args:
            indent: JSON indentation level (None for compact)
            use_api_names: Whether to use API field names
            exclude_unset: Whether to exclude fields that were not set
            exclude_defaults: Whether to exclude fields with default values
            exclude_none: Whether to exclude None values
            warnings: Whether to show warnings
        
        Returns:
            JSON string representation of the response
        """
        data = self.to_dict(
            mode="json",
            use_api_names=use_api_names,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            warnings=warnings,
        )
        return json.dumps(data, indent=indent)
    
    # Aliases for backward compatibility
    def model_dump(self, **kwargs):
        """Alias for to_dict() - Pydantic v2 compatibility."""
        return self.to_dict(**kwargs)
    
    def model_dump_json(self, **kwargs):
        """Alias for to_json() - Pydantic v2 compatibility."""
        return self.to_json(**kwargs)
    
    def dict(self, **kwargs):
        """Deprecated alias for to_dict() - Pydantic v1 compatibility."""
        # Match Pydantic v1 dict() defaults
        kwargs.setdefault('mode', 'python')
        kwargs.setdefault('use_api_names', True)  
        kwargs.setdefault('exclude_unset', False)  # Different default than to_dict()
        kwargs.setdefault('exclude_defaults', False)
        kwargs.setdefault('exclude_none', False)
        return self.to_dict(**kwargs)
    
    def json(self, **kwargs):
        """Deprecated alias for to_json() - Pydantic v1 compatibility."""
        # Match Pydantic v1 json() defaults
        kwargs.setdefault('use_api_names', True)
        kwargs.setdefault('exclude_unset', False)  # Different default than to_json()
        kwargs.setdefault('exclude_defaults', False)
        kwargs.setdefault('exclude_none', False)
        return self.to_json(**kwargs)
    
    def __repr__(self):
        """String representation of the response."""
        attrs = []
        for key, value in self._response_data.items():
            if key.startswith("_"):
                continue
            attrs.append(f"{key}={repr(value)}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def __str__(self):
        """String representation of the response."""
        return self.__repr__()


class _StreamingResponseProxy(_ResponseWrapper):
    """Proxy for OpenAI streaming responses to capture usage and create spans."""
    
    def __init__(self, response, span, tracer, integration, kwargs, start_time, is_async=False):
        # Don't call parent __init__ since we handle initialization differently
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
        self._response_data = {}  # For storing final response data

    async def __aiter__(self):
        """Async iteration for streaming responses."""
        tool_calls_by_index = self._tool_calls_by_index
        stream_obj = getattr(self, "_entered_response", None) or self._response

        try:
            async for chunk in stream_obj:
                self._chunks.append(chunk)
                
                # Accumulate content
                if chunk.choices:
                    for choice in chunk.choices:
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                            self._accumulated_content += choice.delta.content
                        
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
        finally:
            # Ensure span is finished even if iteration is broken early
            # Create tool call spans after streaming completes
            if tool_calls_by_index:
                # Store tool calls in main span attributes
                if 'tool_calls' not in self._span.attributes:
                    self._span.attributes['tool_calls'] = []
                self._span.attributes['tool_calls'].extend(list(tool_calls_by_index.values()))
                
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
            
            self._finish_span(self._accumulated_content)

    def __iter__(self):
        """Sync iteration for streaming responses."""
        tool_calls_by_index = self._tool_calls_by_index
        stream_obj = getattr(self, "_entered_response", None) or self._response

        try:
            for chunk in stream_obj:
                self._chunks.append(chunk)
                
                # Accumulate content
                if chunk.choices:
                    for choice in chunk.choices:
                        if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                            self._accumulated_content += choice.delta.content
                        
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
        finally:
            # Ensure span is finished even if iteration is broken early
            # Create tool call spans after streaming completes
            if tool_calls_by_index:
                # Store tool calls in main span attributes
                if 'tool_calls' not in self._span.attributes:
                    self._span.attributes['tool_calls'] = []
                self._span.attributes['tool_calls'].extend(list(tool_calls_by_index.values()))
                
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
            
            self._finish_span(self._accumulated_content)

    async def __aenter__(self):
        if hasattr(self._response, "__aenter__"):
            self._entered_response = await self._response.__aenter__()
        else:
            self._entered_response = None
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._response, "__aexit__"):
                await self._response.__aexit__(exc_type, exc, tb)
        finally:
            if not self._finished:
                self._finish_span(self._accumulated_content)
        return False

    def __enter__(self):
        if hasattr(self._response, "__enter__"):
            self._entered_response = self._response.__enter__()
        else:
            self._entered_response = None
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if hasattr(self._response, "__exit__"):
                self._response.__exit__(exc_type, exc, tb)
        finally:
            if not self._finished:
                self._finish_span(self._accumulated_content)
        return False
    
    def _finish_span(self, accumulated_content: str):
        """Finish the span with accumulated data."""
        if self._finished:
            return
        self._finished = True
        
        import json
        import time
        
        logger.debug(f"[OpenAI] Finishing streaming span after {len(self._chunks)} chunks, content length: {len(accumulated_content)}")
        
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
        else:
            # Fallback: estimate tokens if usage not available (older API or custom endpoints)
            logger.debug("[OpenAI] No usage data in streaming response, estimating tokens")
            
            # Rough estimation: ~1 token per 4 characters for English text
            if self._kwargs.get("messages"):
                input_chars = sum(len(str(msg.get("content", ""))) for msg in self._kwargs["messages"])
                estimated_input_tokens = max(1, input_chars // 4)
                self._span.attributes["inputTokens"] = estimated_input_tokens
                self._span.attributes["estimated_input_tokens"] = True
            
            if accumulated_content:
                estimated_output_tokens = max(1, len(accumulated_content) // 4)
                self._span.attributes["outputTokens"] = estimated_output_tokens
                self._span.attributes["estimated_output_tokens"] = True
        
        # Set throughput
        throughput = (len(accumulated_content) / elapsed) if (accumulated_content and elapsed > 0) else 0
        self._span.attributes["throughput"] = round(throughput, 2)
        
        # Set I/O data - format tool calls consistently
        output = accumulated_content
        
        # If there are tool calls, include them in a structured format
        if self._tool_calls_by_index:
            tool_calls_data = list(self._tool_calls_by_index.values())
            if tool_calls_data:
                # If no content, show tool calls as primary output
                if not output:
                    output = json.dumps({
                        "tool_calls": tool_calls_data
                    }, indent=2)
                else:
                    # If there's content AND tool calls, show both
                    output = json.dumps({
                        "content": output,
                        "tool_calls": tool_calls_data
                    }, indent=2)
        
        self._span.set_io(
            input_data=json.dumps(self._integration._serialize_messages(self._kwargs.get("messages"))),
            output_data=output
        )
        
        self._tracer.end_span(self._span)
        
        # Populate response data for OpenAI-compatible response methods
        # This mimics the structure of a ChatCompletion response
        self._response_data = {
            "id": getattr(self._chunks[0], 'id', f"chatcmpl-{int(time.time())}") if self._chunks else f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(self._start_time),
            "model": self._kwargs.get("model", "unknown"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": accumulated_content or None,
                    "tool_calls": list(self._tool_calls_by_index.values()) if self._tool_calls_by_index else None
                },
                "finish_reason": self._get_finish_reason()
            }],
            "usage": {
                "prompt_tokens": getattr(self._usage, 'prompt_tokens', 0) if self._usage else 0,
                "completion_tokens": getattr(self._usage, 'completion_tokens', 0) if self._usage else 0,
                "total_tokens": getattr(self._usage, 'total_tokens', 0) if self._usage else 0
            } if self._usage else None
        }
        
        # Set attributes directly on self for attribute access
        for key, value in self._response_data.items():
            setattr(self, key, value)
    
    def _get_finish_reason(self):
        """Extract finish reason from the last chunk."""
        if not self._chunks:
            return "stop"
        
        for chunk in reversed(self._chunks):
            if hasattr(chunk, 'choices') and chunk.choices:
                for choice in chunk.choices:
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        return choice.finish_reason
        
        # Default to "stop" if no finish reason found
        return "stop"
    
    def _build_response_data(self):
        """Build response data from current state."""
        import time
        
        if not self._response_data:
            # Build response data on demand if not already built
            self._response_data = {
                "id": getattr(self._chunks[0], 'id', f"chatcmpl-{int(time.time())}") if self._chunks else f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(self._start_time),
                "model": self._kwargs.get("model", "unknown"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self._accumulated_content or None,
                        "tool_calls": list(self._tool_calls_by_index.values()) if self._tool_calls_by_index else None
                    },
                    "finish_reason": self._get_finish_reason() if self._finished else None
                }],
                "usage": {
                    "prompt_tokens": getattr(self._usage, 'prompt_tokens', 0) if self._usage else 0,
                    "completion_tokens": getattr(self._usage, 'completion_tokens', 0) if self._usage else 0,
                    "total_tokens": getattr(self._usage, 'total_tokens', 0) if self._usage else 0
                } if self._usage else None
            }
        return self._response_data
    
    # Override parent methods to ensure response data is available
    def to_dict(self, **kwargs):
        """Generate a dictionary representation of the response."""
        self._build_response_data()
        return super().to_dict(**kwargs)
    
    def to_json(self, **kwargs):
        """Generate a JSON string representation of the response."""
        self._build_response_data()
        return super().to_json(**kwargs)
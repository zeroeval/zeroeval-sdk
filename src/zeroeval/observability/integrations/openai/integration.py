import json
import logging
import re
import os
from functools import wraps
from typing import Any, Callable, Optional

from ..base import Integration

logger = logging.getLogger(__name__)


def extract_pydantic_schema_for_openai(model_type: Any) -> Optional[dict[str, Any]]:
    """
    Extract Pydantic schema in OpenAI's response_format structure.
    
    Returns a dict that can be directly used as the response_format parameter
    in openai.chat.completions.create calls.
    """
    try:
        # Check if it's a Pydantic model
        import pydantic
        
        if not (isinstance(model_type, type) and issubclass(model_type, pydantic.BaseModel)):
            logger.debug(f"Type {model_type} is not a Pydantic BaseModel")
            return None
            
        # Get the JSON schema from the Pydantic model
        if hasattr(model_type, 'model_json_schema'):
            # Pydantic v2
            schema = model_type.model_json_schema()
        elif hasattr(model_type, 'schema'):
            # Pydantic v1
            schema = model_type.schema()
        else:
            logger.warning(f"Unable to extract schema from {model_type}")
            return None
            
        # Return in OpenAI's response_format structure
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model_type.__name__,
                "schema": schema,
                "strict": True
            }
        }
        
    except ImportError:
        logger.debug("Pydantic not installed, cannot extract model schema")
        return None
    except Exception as e:
        logger.warning(f"Error extracting Pydantic model schema: {e}")
        return None


def zeroeval_prompt(
    name: str,
    content: str,
    variables: Optional[dict] = None,
    *,
    prompt_slug: Optional[str] = None,
    prompt_version: Optional[int] = None,
    prompt_version_id: Optional[str] = None,
    content_hash: Optional[str] = None,
) -> str:
    """
    Helper function to create a prompt with zeroeval metadata for tracing and observability.
    
    When this prompt is used in an OpenAI API call, ZeroEval will automatically:
    1. Extract the task metadata from the prompt
    2. Link the span to the specified task
    3. Create the task automatically if it doesn't exist yet
    
    Args:
        name: Required task identifier for this prompt
        content: The actual prompt content (e.g., "You are a helpful assistant.")
        variables: Optional dictionary of variables to be interpolated in the prompt
    
    Returns:
        A string with the format: <zeroeval>{JSON}</zeroeval>content
        
    Example:
        >>> zeroeval_prompt(
        ...     name="custom-bot-5",
        ...     content="You are an assistant that helps users with {{task}}. Be {{tone}} in your responses.",
        ...     variables={
        ...         "task": "coding questions",
        ...         "tone": "helpful and concise"
        ...     }
        ... )
        '<zeroeval>{"task": "custom-bot-5", "variables": {"task": "coding questions", "tone": "helpful and concise"}}</zeroeval>You are an assistant that helps users with {{task}}. Be {{tone}} in your responses.'
        
    Note:
        - Variables will be interpolated in the prompt when the OpenAI API is called
        - The task will be automatically created in ZeroEval if it doesn't exist
        - The returned string MUST be included in your OpenAI messages for tuning span linkage to work
    """
    logger.info(f"=== zeroeval_prompt() called ===")
    logger.info(f"  task name: '{name}'")
    logger.debug(f"  content length: {len(content)} chars")
    logger.debug(f"  variables: {list(variables.keys()) if variables else None}")
    logger.debug(f"  prompt_slug: {prompt_slug}")
    logger.debug(f"  prompt_version: {prompt_version}")
    logger.debug(f"  prompt_version_id: {prompt_version_id}")
    logger.debug(f"  content_hash: {content_hash}")
    
    metadata = {"task": name}
    
    if variables:
        metadata["variables"] = variables
        logger.debug(f"zeroeval_prompt: Adding {len(variables)} variables to metadata")
    
    # Optional prompt linkage metadata - these are CRITICAL for tuning span creation
    if prompt_slug:
        metadata["prompt_slug"] = prompt_slug
    if prompt_version is not None:
        metadata["prompt_version"] = int(prompt_version)
    if prompt_version_id:
        metadata["prompt_version_id"] = str(prompt_version_id)
        logger.info(f"zeroeval_prompt: prompt_version_id={prompt_version_id} - this enables tuning span linkage")
    else:
        logger.warning(f"zeroeval_prompt: No prompt_version_id provided - tuning span linkage may not work!")
    if content_hash:
        metadata["content_hash"] = str(content_hash)
    
    metadata_json = json.dumps(metadata)
    formatted_prompt = f'<zeroeval>{metadata_json}</zeroeval>{content}'
    
    logger.info(f"=== zeroeval_prompt() result ===")
    logger.info(f"  Metadata embedded: task='{name}', version_id={prompt_version_id}, version={prompt_version}")
    logger.debug(f"  Full metadata JSON: {metadata_json}")
    logger.debug(f"  Formatted prompt length: {len(formatted_prompt)} chars")
    logger.debug(f"  Formatted prompt preview: {formatted_prompt[:150]}..." if len(formatted_prompt) > 150 else f"  Formatted prompt: {formatted_prompt}")
    logger.info(f"  IMPORTANT: This string must be included in your OpenAI messages for tuning to work!")
    
    return formatted_prompt


def get_provider_from_base_url(base_url: Optional[str]) -> str:
    """
    Determine the provider based on the base URL.

    Args:
        base_url: The base URL of the OpenAI-compatible API

    Returns:
        The provider name (e.g., "openai", "zeroeval", "novita", etc.)
    """
    if not base_url:
        return "openai"

    base_url = base_url.lower()

    # Map of URL patterns to provider names
    provider_mapping = {
        "localhost": "zeroeval",
        "127.0.0.1": "zeroeval",
        "api.zeroeval.com": "zeroeval",
        "api.openai.com": "openai",
        "api.novita.ai": "novita",
        "api.deepinfra.com": "deepinfra",
        "api.llm-stats.com": "llm-stats",
    }

    # Check each pattern in the base URL
    for pattern, provider in provider_mapping.items():
        if pattern in base_url:
            return provider

    # Default to "openai" if no match found
    return "openai"


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
            # Also patch Azure client variants if available
            try:
                if hasattr(openai, "AzureOpenAI"):
                    self._patch_method(
                        openai.AzureOpenAI,
                        "__init__",
                        self._wrap_init
                    )
                if hasattr(openai, "AsyncAzureOpenAI"):
                    self._patch_method(
                        openai.AsyncAzureOpenAI,
                        "__init__",
                        self._wrap_init
                    )
            except Exception as e:
                logger.debug(f"Azure OpenAI classes not fully patched: {e}")
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
                    # Patch responses.parse as well
                    self._patch_method(
                        client_instance.responses,
                        "parse",
                        self._wrap_responses_parse_async()
                    )
                else:
                    self._patch_method(
                        client_instance.responses,
                        "create",
                        self._wrap_responses_create_sync(client_instance.responses)
                    )
                    # Patch responses.parse as well
                    self._patch_method(
                        client_instance.responses,
                        "parse",
                        self._wrap_responses_parse_sync(client_instance.responses)
                    )
            
            return result
        return wrapper

    def _extract_zeroeval_metadata(self, content: str) -> tuple[Optional[dict[str, Any]], str]:
        """
        Extract <zeroeval> metadata from content and return (metadata, cleaned_content).
        
        Returns:
            - Tuple of (metadata dict or None, cleaned content string)
        """
        logger.debug(f"_extract_zeroeval_metadata: Searching for metadata in content (length: {len(content)} chars)")
        logger.debug(f"_extract_zeroeval_metadata: Content preview: {content[:200]}..." if len(content) > 200 else f"_extract_zeroeval_metadata: Content: {content}")
        
        # Look for <zeroeval>...</zeroeval> tags
        pattern = r'<zeroeval>(.*?)</zeroeval>'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            logger.debug("_extract_zeroeval_metadata: No <zeroeval> tags found in content")
            # Additional debug: check if there's a partial match that might indicate an issue
            if "<zeroeval" in content:
                logger.warning("_extract_zeroeval_metadata: Found '<zeroeval' but no complete tag - possible malformed tag")
            if "</zeroeval>" in content:
                logger.warning("_extract_zeroeval_metadata: Found '</zeroeval>' but no opening tag")
            return None, content
        
        logger.info("_extract_zeroeval_metadata: Found <zeroeval> tags, extracting metadata")
        
        try:
            # Extract and parse the JSON
            json_str = match.group(1).strip()
            logger.debug(f"_extract_zeroeval_metadata: Raw JSON string: {json_str}")
            
            metadata = json.loads(json_str)
            
            # Validate required fields
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a JSON object")
            
            # Log ALL metadata fields important for tuning span creation
            logger.info(f"_extract_zeroeval_metadata: === PARSED METADATA ===")
            logger.info(f"  task: {metadata.get('task', 'NOT SET')}")
            logger.info(f"  prompt_version: {metadata.get('prompt_version', 'NOT SET')}")
            logger.info(f"  prompt_version_id: {metadata.get('prompt_version_id', 'NOT SET')}")
            logger.info(f"  prompt_slug: {metadata.get('prompt_slug', 'NOT SET')}")
            logger.info(f"  content_hash: {metadata.get('content_hash', 'NOT SET')}")
            if "variables" in metadata:
                logger.debug(f"  variables: {list(metadata['variables'].keys())}")
            
            # Warn if critical fields for tuning are missing
            if not metadata.get('task'):
                logger.warning("_extract_zeroeval_metadata: 'task' field is missing - tuning span will NOT be created!")
            if not metadata.get('prompt_version_id'):
                logger.warning("_extract_zeroeval_metadata: 'prompt_version_id' is missing - tuning span linkage may fail!")
            
            # Remove the <zeroeval> tags from content
            cleaned_content = re.sub(pattern, '', content, count=1).strip()
            logger.debug(f"_extract_zeroeval_metadata: Cleaned content length: {len(cleaned_content)} chars")
            
            return metadata, cleaned_content
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"_extract_zeroeval_metadata: Failed to parse metadata JSON: {e}")
            logger.error(f"_extract_zeroeval_metadata: Invalid JSON string was: {json_str if 'json_str' in locals() else 'N/A'}")
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
    
    def _log_task_metadata(self, task_id: Optional[str], zeroeval_metadata: dict[str, Any], context: str = "OpenAI API call") -> None:
        """
        Log information about task metadata found in zeroeval_prompt.
        
        Args:
            task_id: The task ID from metadata
            zeroeval_metadata: Full metadata dictionary
            context: Context string for the log message
        """
        if task_id:
            logger.info(
                f"{context}: Task ID '{task_id}' added to span attributes. "
                f"The task will be automatically created if it doesn't exist yet, "
                f"and this span will be linked to it for tracing and tuning."
            )
        logger.debug(f"{context}: Full zeroeval metadata added to span: {zeroeval_metadata}")
    
    def _process_messages_with_zeroeval(self, messages: Optional[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """
        Process messages to extract zeroeval metadata and interpolate variables.
        
        This function searches ALL messages for <zeroeval> tags, not just the first
        system message. This ensures ze.prompt() works regardless of:
        - Whether the prompt is in a system or user message
        - The position of the message in the array
        - Whether the message is the first one or not
        
        Returns:
            - Tuple of (processed messages, zeroeval metadata)
        """
        if not messages:
            logger.debug("_process_messages_with_zeroeval: No messages to process")
            return messages, None
        
        logger.info(f"_process_messages_with_zeroeval: Processing {len(messages)} messages")
        logger.debug(f"_process_messages_with_zeroeval: Message roles: {[m.get('role') for m in messages]}")
        
        # Deep copy messages to avoid modifying the original
        import copy
        processed_messages = copy.deepcopy(messages)
        zeroeval_metadata = None
        variables = {}
        metadata_found_in_message_idx = None
        
        # Search ALL messages for zeroeval tags (not just first system message)
        for i, msg in enumerate(processed_messages):
            content = msg.get("content", "")
            role = msg.get("role", "unknown")
            
            # Skip if no content
            if not content:
                logger.debug(f"_process_messages_with_zeroeval: Message {i} ({role}) has no content, skipping")
                continue
            
            # Check if this message contains <zeroeval> tags
            if "<zeroeval>" in content:
                logger.info(f"_process_messages_with_zeroeval: Found <zeroeval> tag in message {i} (role: {role})")
                
                # Extract zeroeval metadata
                metadata, cleaned_content = self._extract_zeroeval_metadata(content)
                
                if metadata:
                    if zeroeval_metadata is not None:
                        # Already found metadata in a previous message - warn but use the first one
                        logger.warning(
                            f"_process_messages_with_zeroeval: Multiple <zeroeval> tags found! "
                            f"Using metadata from message {metadata_found_in_message_idx}, ignoring message {i}"
                        )
                    else:
                        zeroeval_metadata = metadata
                        variables = metadata.get("variables", {})
                        metadata_found_in_message_idx = i
                        
                        # Update this message with cleaned content
                        processed_messages[i]["content"] = cleaned_content
                        
                        # Log extraction details
                        task_id = metadata.get('task')
                        prompt_version = metadata.get('prompt_version')
                        prompt_version_id = metadata.get('prompt_version_id')
                        content_hash = metadata.get('content_hash')
                        prompt_slug = metadata.get('prompt_slug')
                        
                        logger.info(f"_process_messages_with_zeroeval: === ZEROEVAL METADATA EXTRACTED ===")
                        logger.info(f"  - Found in: message {i} (role: {role})")
                        logger.info(f"  - Task ID: '{task_id}'")
                        logger.info(f"  - Prompt Version: {prompt_version}")
                        logger.info(f"  - Prompt Version ID: {prompt_version_id}")
                        logger.info(f"  - Content Hash: {content_hash}")
                        logger.info(f"  - Prompt Slug: {prompt_slug}")
                        logger.info(f"  - Variables: {list(variables.keys()) if variables else 'none'}")
                        
                        # Log task linkage info
                        if task_id:
                            logger.info(
                                f"_process_messages_with_zeroeval: Task '{task_id}' will be linked to this span. "
                                f"Tuning span should be created with version_id={prompt_version_id}"
                            )
                        else:
                            logger.warning(
                                f"_process_messages_with_zeroeval: No task ID in metadata - tuning span linking may fail!"
                            )
            else:
                logger.debug(f"_process_messages_with_zeroeval: Message {i} ({role}) has no <zeroeval> tags")
        
        # Log if no metadata was found
        if zeroeval_metadata is None:
            logger.warning(
                f"_process_messages_with_zeroeval: No <zeroeval> metadata found in any of the {len(messages)} messages. "
                f"If you used ze.prompt(), ensure the returned string is included in your messages. "
                f"Message contents preview: {[m.get('content', '')[:50] + '...' if m.get('content') and len(m.get('content', '')) > 50 else m.get('content', '') for m in messages]}"
            )
        
        # Interpolate variables in all messages if we have any
        if variables:
            logger.debug(f"_process_messages_with_zeroeval: Interpolating {len(variables)} variables in all messages")
            for i, msg in enumerate(processed_messages):
                if msg.get("content"):
                    original_content = msg["content"]
                    msg["content"] = self._interpolate_variables(msg["content"], variables)
                    if original_content != msg["content"]:
                        logger.debug(f"_process_messages_with_zeroeval: Variables interpolated in message {i}")
        
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

    def _process_responses_input_with_zeroeval(self, input_data: Any) -> tuple[Any, list[dict[str, Any]], Optional[dict[str, Any]]]:
        """
        Process Responses API `input` to support zeroeval_prompt metadata and variable interpolation.

        Returns:
          - processed_input: original input shape with <zeroeval> removed and variables interpolated
          - normalized_messages: best-effort normalized messages for span I/O
          - zeroeval_metadata: extracted metadata dict or None
        """
        import copy as _copy

        zeroeval_metadata: Optional[dict[str, Any]] = None
        variables: dict[str, Any] = {}

        def _process_text(text: Optional[str]) -> Optional[str]:
            nonlocal zeroeval_metadata, variables
            if not isinstance(text, str):
                return text
            if zeroeval_metadata is None:
                meta, cleaned = self._extract_zeroeval_metadata(text)
                if meta:
                    zeroeval_metadata = meta
                    variables = meta.get("variables", {}) or {}
                    text = cleaned
            if variables:
                text = self._interpolate_variables(text, variables)
            return text

        # String input
        if isinstance(input_data, str):
            processed = _process_text(input_data)
            return processed, self._serialize_responses_input(processed), zeroeval_metadata

        # List input (messages or content parts)
        if isinstance(input_data, list):
            processed_list = _copy.deepcopy(input_data)
            for item in processed_list:
                if isinstance(item, dict):
                    # Message-like
                    if item.get("role") is not None and item.get("content") is not None:
                        content = item.get("content")
                        if isinstance(content, str):
                            item["content"] = _process_text(content)
                        elif isinstance(content, list):
                            new_parts = []
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    part = {**part, "text": _process_text(part.get("text"))}
                                new_parts.append(part)
                            item["content"] = new_parts
                        continue

                    # input_text-style
                    if item.get("type") in ("input_text", "text") and item.get("text") is not None:
                        item["text"] = _process_text(item.get("text"))
                        continue
            return processed_list, self._serialize_responses_input(processed_list), zeroeval_metadata

        # Dict or other types: cannot safely mutate; process via normalized messages
        normalized = self._serialize_responses_input(input_data)
        if normalized and isinstance(normalized[0].get("content"), str):
            normalized[0]["content"] = _process_text(normalized[0]["content"])  # type: ignore[index]
        return input_data, normalized, zeroeval_metadata

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
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "streaming": is_streaming,
                    "base_url": base_url,
                }
                
                # Always capture tools and tool_choice if present in the request
                if "tools" in kwargs and kwargs["tools"]:
                    # Store tool definitions for rendering
                    tools_info = []
                    for tool in kwargs["tools"]:
                        # Handle both dict and Pydantic objects
                        if hasattr(tool, "model_dump"):
                            tool_dict = tool.model_dump()
                        elif hasattr(tool, "dict"):
                            tool_dict = tool.dict()
                        else:
                            tool_dict = tool

                        if tool_dict.get("type") == "function" and "function" in tool_dict:
                            tools_info.append({
                                "type": "function",
                                "name": tool_dict["function"].get("name"),
                                "description": tool_dict["function"].get("description")
                            })
                    span_attributes["tools"] = tools_info
                    
                    # Convert tools to serializable format for storage
                    try:
                        tools_raw = [
                            t.model_dump() if hasattr(t, "model_dump") else (t.dict() if hasattr(t, "dict") else t)
                            for t in kwargs["tools"]
                        ]
                        span_attributes["tools_raw"] = tools_raw
                    except Exception:
                        pass
                # Back-compat: capture legacy 'functions' param as tools metadata
                if "functions" in kwargs and kwargs["functions"] and not span_attributes.get("tools"):
                    try:
                        legacy_tools = []
                        for fn in kwargs["functions"]:
                            # fn expected shape: { name, description, parameters }
                            legacy_tools.append({
                                "type": "function",
                                "name": fn.get("name"),
                                "description": fn.get("description")
                            })
                        if legacy_tools:
                            span_attributes["tools"] = legacy_tools
                            span_attributes["tools_raw_functions"] = kwargs["functions"]
                            # Reflect tool_choice if using legacy flow
                            if "function_call" in kwargs:
                                span_attributes["tool_choice"] = kwargs["function_call"]
                    except Exception:
                        pass
                
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                
                # Add zeroeval metadata to attributes if present
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    # Attach full zeroeval metadata so backend can read prompt_version_id, prompt_slug, etc.
                    span_attributes["zeroeval"] = zeroeval_metadata
                    # Attempt to patch model from prompt_version_id if present
                    try:
                        prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                        if prompt_version_id:
                            from ....client import ZeroEval as _PromptClient
                            _pc = _PromptClient()
                            _model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                            if _model:
                                kwargs["model"] = _model
                    except Exception:
                        pass
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI chat.completions.create")
                    
                    # Store the original system prompt template (with {{variables}})
                    if original_messages and original_messages[0].get("role") == "system":
                        # Extract just the content after the zeroeval tags
                        _, template_content = self._extract_zeroeval_metadata(original_messages[0].get("content", ""))
                        if template_content:
                            span_attributes["system_prompt_template"] = template_content
                    
                # Ensure we record the resolved model/provider after any ZeroEval prompt resolution
                _span_model = kwargs.get("model")
                if isinstance(_span_model, str) and _span_model.startswith("zeroeval/"):
                    span_attributes["provider"] = "zeroeval"
                    _span_model = _span_model.split("/", 1)[1]
                span_attributes["model"] = _span_model
                span = self.tracer.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai"},
                )
                tracer = self.tracer

                # Temporarily route via ZeroEval proxy if model is a ZeroEval-prefixed id
                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        span.attributes["provider"] = "zeroeval"
                        
                        if args and hasattr(args[0], "_client"):
                            # For resources like chat.completions that have a _client
                            if hasattr(args[0]._client, "base_url"):
                                original_base_url = args[0]._client.base_url
                                args[0]._client.base_url = ze_base
                            if ze_api_key and hasattr(args[0]._client, "api_key"):
                                original_api_key = args[0]._client.api_key
                                args[0]._client.api_key = ze_api_key
                        elif args and hasattr(args[0], "base_url"):
                            # For direct client instances
                            original_base_url = args[0].base_url
                            args[0].base_url = ze_base
                            if ze_api_key and hasattr(args[0], "api_key"):
                                original_api_key = args[0].api_key
                                args[0].api_key = ze_api_key

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
                finally:
                    try:
                        if args and hasattr(args[0], "_client"):
                            if original_base_url is not None and hasattr(args[0]._client, "base_url"):
                                args[0]._client.base_url = original_base_url
                            if original_api_key is not None and hasattr(args[0]._client, "api_key"):
                                args[0]._client.api_key = original_api_key
                        elif args:
                            if original_base_url is not None and hasattr(args[0], "base_url"):
                                args[0].base_url = original_base_url
                            if original_api_key is not None and hasattr(args[0], "api_key"):
                                args[0].api_key = original_api_key
                    except Exception:
                        pass

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
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "streaming": is_streaming,
                    "base_url": base_url,
                }
                
                # Always capture tools and tool_choice if present in the request
                if "tools" in kwargs and kwargs["tools"]:
                    # Store tool definitions for rendering
                    tools_info = []
                    for tool in kwargs["tools"]:
                        # Handle both dict and Pydantic objects
                        if hasattr(tool, "model_dump"):
                            tool_dict = tool.model_dump()
                        elif hasattr(tool, "dict"):
                            tool_dict = tool.dict()
                        else:
                            tool_dict = tool

                        if tool_dict.get("type") == "function" and "function" in tool_dict:
                            tools_info.append({
                                "type": "function",
                                "name": tool_dict["function"].get("name"),
                                "description": tool_dict["function"].get("description")
                            })
                    span_attributes["tools"] = tools_info
                    
                    # Convert tools to serializable format for storage
                    try:
                        tools_raw = [
                            t.model_dump() if hasattr(t, "model_dump") else (t.dict() if hasattr(t, "dict") else t)
                            for t in kwargs["tools"]
                        ]
                        span_attributes["tools_raw"] = tools_raw
                    except Exception:
                        pass
                # Back-compat: capture legacy 'functions' param as tools metadata
                if "functions" in kwargs and kwargs["functions"] and not span_attributes.get("tools"):
                    try:
                        legacy_tools = []
                        for fn in kwargs["functions"]:
                            legacy_tools.append({
                                "type": "function",
                                "name": fn.get("name"),
                                "description": fn.get("description")
                            })
                        if legacy_tools:
                            span_attributes["tools"] = legacy_tools
                            span_attributes["tools_raw_functions"] = kwargs["functions"]
                            if "function_call" in kwargs:
                                span_attributes["tool_choice"] = kwargs["function_call"]
                    except Exception:
                        pass
                
                # Capture tool_choice if present
                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]
                
                # Add zeroeval metadata to attributes if present
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    span_attributes["zeroeval"] = zeroeval_metadata
                    # Attempt to patch model via content_hash+task or version_id
                    try:
                        from ....client import ZeroEval as _PromptClient
                        _pc = _PromptClient()
                        _patched_model = None
                        task_name = zeroeval_metadata.get("task")
                        chash = zeroeval_metadata.get("content_hash")
                        if task_name and chash:
                            try:
                                prompt_obj = _pc.get_task_prompt_version_by_hash(task_name=task_name, content_hash=chash)
                                if getattr(prompt_obj, "model", None):
                                    _patched_model = prompt_obj.model
                            except Exception:
                                pass
                        if not _patched_model:
                            prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                            if prompt_version_id:
                                _patched_model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                        if _patched_model:
                            kwargs["model"] = _patched_model
                    except Exception:
                        pass
                    # Attempt to patch model from prompt_version_id if present
                    try:
                        prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                        if prompt_version_id:
                            from ....client import ZeroEval as _PromptClient
                            _pc = _PromptClient()
                            _model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                            if _model:
                                kwargs["model"] = _model
                    except Exception:
                        pass
                    # Attempt to patch model from prompt_version_id if present
                    try:
                        prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                        if prompt_version_id:
                            from ....client import ZeroEval as _PromptClient
                            _pc = _PromptClient()
                            _model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                            if _model:
                                kwargs["model"] = _model
                    except Exception:
                        pass
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI chat.completions.create")
                    
                    # Store the original system prompt template (with {{variables}})
                    if original_messages and original_messages[0].get("role") == "system":
                        # Extract just the content after the zeroeval tags
                        _, template_content = self._extract_zeroeval_metadata(original_messages[0].get("content", ""))
                        if template_content:
                            span_attributes["system_prompt_template"] = template_content
                    
                # Ensure we record the resolved model/provider after any ZeroEval prompt resolution
                _span_model = kwargs.get("model")
                if isinstance(_span_model, str) and _span_model.startswith("zeroeval/"):
                    span_attributes["provider"] = "zeroeval"
                    _span_model = _span_model.split("/", 1)[1]
                span_attributes["model"] = _span_model
                span = self.tracer.start_span(
                    name="openai.chat.completions.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai"},
                )
                tracer = self.tracer

                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        span.attributes["provider"] = "zeroeval"
                        
                        if hasattr(completions_instance, "_client"):
                            if hasattr(completions_instance._client, "base_url"):
                                original_base_url = completions_instance._client.base_url
                                completions_instance._client.base_url = ze_base
                            if ze_api_key and hasattr(completions_instance._client, "api_key"):
                                original_api_key = completions_instance._client.api_key
                                completions_instance._client.api_key = ze_api_key
                        elif hasattr(completions_instance, "base_url"):
                            original_base_url = completions_instance.base_url
                            completions_instance.base_url = ze_base
                            if ze_api_key and hasattr(completions_instance, "api_key"):
                                original_api_key = completions_instance.api_key
                                completions_instance.api_key = ze_api_key

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
                finally:
                    try:
                        if hasattr(completions_instance, "_client"):
                            if original_base_url is not None and hasattr(completions_instance._client, "base_url"):
                                completions_instance._client.base_url = original_base_url
                            if original_api_key is not None and hasattr(completions_instance._client, "api_key"):
                                completions_instance._client.api_key = original_api_key
                        else:
                            if original_base_url is not None and hasattr(completions_instance, "base_url"):
                                completions_instance.base_url = original_base_url
                            if original_api_key is not None and hasattr(completions_instance, "api_key"):
                                completions_instance.api_key = original_api_key
                    except Exception:
                        pass
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
                
                # Extract input, process zeroeval metadata, and normalize to messages
                raw_input = kwargs.get("input")
                processed_input, normalized_messages, zeroeval_metadata = self._process_responses_input_with_zeroeval(raw_input)
                # Replace input with processed version so the model receives cleaned/interpolated text
                if processed_input is not None:
                    kwargs["input"] = processed_input

                # Try to get base_url from client instance
                base_url = None
                if args and hasattr(args[0], 'base_url'):
                    base_url = str(args[0].base_url)

                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "endpoint": "responses",
                    "base_url": base_url,
                }
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    span_attributes["zeroeval"] = zeroeval_metadata
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI responses")
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
                
                # Ensure we record the resolved model/provider prior to span creation
                _span_model = kwargs.get("model")
                if isinstance(_span_model, str) and _span_model.startswith("zeroeval/"):
                    span_attributes["provider"] = "zeroeval"
                    _span_model = _span_model.split("/", 1)[1]
                span_attributes["model"] = _span_model
                span = self.tracer.start_span(
                    name="openai.responses.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses"},
                )
                tracer = self.tracer

                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        span.attributes["provider"] = "zeroeval"
                        
                        if args and hasattr(args[0], "_client"):
                            if hasattr(args[0]._client, "base_url"):
                                original_base_url = args[0]._client.base_url
                                args[0]._client.base_url = ze_base
                            if ze_api_key and hasattr(args[0]._client, "api_key"):
                                original_api_key = args[0]._client.api_key
                                args[0]._client.api_key = ze_api_key
                        elif args and hasattr(args[0], "base_url"):
                            original_base_url = args[0].base_url
                            args[0].base_url = ze_base
                            if ze_api_key and hasattr(args[0], "api_key"):
                                original_api_key = args[0].api_key
                                args[0].api_key = ze_api_key

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
                
                # Extract input, process zeroeval metadata, and normalize
                raw_input = kwargs.get("input")
                processed_input, normalized_messages, zeroeval_metadata = self._process_responses_input_with_zeroeval(raw_input)
                if processed_input is not None:
                    kwargs["input"] = processed_input

                # Try to get base_url from client instance
                base_url = None
                if args and hasattr(args[0], 'base_url'):
                    base_url = str(args[0].base_url)

                # Prepare span attributes
                span_attributes = {
                    "service.name": "openai",
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "endpoint": "responses",
                    "base_url": base_url,
                }
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    span_attributes["zeroeval"] = zeroeval_metadata
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI responses")
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
                
                # Ensure we record the resolved model/provider prior to span creation
                _span_model = kwargs.get("model")
                if isinstance(_span_model, str) and _span_model.startswith("zeroeval/"):
                    span_attributes["provider"] = "zeroeval"
                    _span_model = _span_model.split("/", 1)[1]
                span_attributes["model"] = _span_model
                span = self.tracer.start_span(
                    name="openai.responses.create",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses"},
                )
                tracer = self.tracer

                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        span.attributes["provider"] = "zeroeval"
                        
                        if hasattr(responses_instance, "_client"):
                            if hasattr(responses_instance._client, "base_url"):
                                original_base_url = responses_instance._client.base_url
                                responses_instance._client.base_url = ze_base
                            if ze_api_key and hasattr(responses_instance._client, "api_key"):
                                original_api_key = responses_instance._client.api_key
                                responses_instance._client.api_key = ze_api_key
                        elif hasattr(responses_instance, "base_url"):
                            original_base_url = responses_instance.base_url
                            responses_instance.base_url = ze_base
                            if ze_api_key and hasattr(responses_instance, "api_key"):
                                original_api_key = responses_instance.api_key
                                responses_instance.api_key = ze_api_key

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

    # ------------------------------------------------------------------+
    #  Responses API parse wrappers                                      |
    # ------------------------------------------------------------------+
    def _wrap_responses_parse_async(self) -> Callable:
        """Wrap responses.parse for async clients."""
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            async def inner(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                # Process input to support zeroeval metadata and variable interpolation
                raw_input = kwargs.get("input")
                processed_input, normalized_messages, zeroeval_metadata = self._process_responses_input_with_zeroeval(raw_input)
                if processed_input is not None:
                    kwargs["input"] = processed_input

                # Try to get base_url from client instance
                base_url = None
                if args and hasattr(args[0], 'base_url'):
                    base_url = str(args[0].base_url)

                span_attributes = {
                    "service.name": "openai",
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "endpoint": "responses.parse",
                    "base_url": base_url,
                }
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    span_attributes["zeroeval"] = zeroeval_metadata
                    
                    # Attempt to patch model from prompt_version_id if present
                    try:
                        prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                        if prompt_version_id:
                            from ....client import ZeroEval as _PromptClient
                            _pc = _PromptClient()
                            _model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                            if _model:
                                kwargs["model"] = _model
                    except Exception:
                        pass
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI responses")

                # base_url if available (resource instance is args[0])
                base_url = None
                if args and hasattr(args[0], "_client") and hasattr(args[0]._client, "base_url"):
                    try:
                        base_url = str(args[0]._client.base_url)
                    except Exception:
                        base_url = None
                if base_url:
                    span_attributes["base_url"] = base_url

                if "tools" in kwargs and kwargs["tools"]:
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if isinstance(tool, dict) and tool.get("type") == "function":
                            tools_info.append({
                                "type": "function",
                                "name": tool.get("function", {}).get("name"),
                                "description": tool.get("function", {}).get("description"),
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]

                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]

                # Capture text_format info with full Pydantic schema in OpenAI format
                if "text_format" in kwargs and kwargs["text_format"] is not None:
                    try:
                        # Extract Pydantic schema in OpenAI's response_format structure
                        response_format = extract_pydantic_schema_for_openai(kwargs["text_format"])
                        if response_format:
                            span_attributes["text_format_response_format"] = response_format
                            # Also keep the simple name for backward compatibility
                            span_attributes["text_format"] = response_format["json_schema"]["name"]
                        else:
                            # Fallback to simple name if not a Pydantic model
                            span_attributes["text_format"] = getattr(kwargs["text_format"], "__name__", str(kwargs["text_format"]))
                    except Exception as e:
                        logger.debug(f"Error capturing text_format: {e}")
                        span_attributes["text_format"] = "<unknown>"

                if normalized_messages:
                    span_attributes["messages"] = normalized_messages

                span = self.tracer.start_span(
                    name="openai.responses.parse",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses.parse"},
                )
                tracer = self.tracer

                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        
                        if args and hasattr(args[0], "_client"):
                            if hasattr(args[0]._client, "base_url"):
                                original_base_url = args[0]._client.base_url
                                args[0]._client.base_url = ze_base
                            if ze_api_key and hasattr(args[0]._client, "api_key"):
                                original_api_key = args[0]._client.api_key
                                args[0]._client.api_key = ze_api_key
                        elif args and hasattr(args[0], "base_url"):
                            original_base_url = args[0].base_url
                            args[0].base_url = ze_base
                            if ze_api_key and hasattr(args[0], "api_key"):
                                original_api_key = args[0].api_key
                                args[0].api_key = ze_api_key

                    response = await original(*args, **kwargs)

                    elapsed = time.time() - start_time

                    usage = getattr(response, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                        span.attributes.update({
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                        })
                        span.attributes["usage"] = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": getattr(usage, "total_tokens", input_tokens + output_tokens),
                        }

                    output_text = getattr(response, "output_text", "")
                    if hasattr(response, "id"):
                        span.attributes["openai_id"] = response.id
                    if hasattr(response, "system_fingerprint"):
                        span.attributes["system_fingerprint"] = response.system_fingerprint

                    # Extract tool calls and reasoning from parsed response output
                    output_items = getattr(response, "output", [])
                    tool_calls = []
                    reasoning_trace = []
                    for item in output_items:
                        if hasattr(item, "type"):
                            if item.type == "function_call":
                                tool_call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {"name": getattr(item, "name", None), "arguments": getattr(item, "arguments", "")},
                                })
                            elif item.type == "reasoning" and hasattr(item, "summary"):
                                for summary in item.summary:
                                    if hasattr(summary, "text"):
                                        reasoning_trace.append(summary.text)

                    if tool_calls:
                        span.attributes["tool_calls"] = tool_calls
                        try:
                            existing_messages = span.attributes.get("messages") or normalized_messages or []
                        except Exception:
                            existing_messages = normalized_messages or []
                        try:
                            messages_with_assistant = list(existing_messages)
                        except Exception:
                            messages_with_assistant = []
                        messages_with_assistant.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                        span.attributes["messages"] = messages_with_assistant
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

                    throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)

                    combined_output = output_text if output_text else (None if tool_calls else None)
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(normalized_messages)),
                        output_data=combined_output,
                    )

                    tracer.end_span(span)
                    if isinstance(response, dict) and not hasattr(response, "to_dict"):
                        return _ResponseWrapper(response)
                    return response
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise
                finally:
                    try:
                        if args and hasattr(args[0], "_client"):
                            if original_base_url is not None and hasattr(args[0]._client, "base_url"):
                                args[0]._client.base_url = original_base_url
                            if original_api_key is not None and hasattr(args[0]._client, "api_key"):
                                args[0]._client.api_key = original_api_key
                        elif args:
                            if original_base_url is not None and hasattr(args[0], "base_url"):
                                args[0].base_url = original_base_url
                            if original_api_key is not None and hasattr(args[0], "api_key"):
                                args[0].api_key = original_api_key
                    except Exception:
                        pass

            return inner
        return wrapper

    def _wrap_responses_parse_sync(self, responses_instance) -> Callable:
        """Wrap responses.parse for sync clients."""
        import json
        import time

        def wrapper(original: Callable) -> Callable:
            @wraps(original)
            def inner(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                raw_input = kwargs.get("input")
                processed_input, normalized_messages, zeroeval_metadata = self._process_responses_input_with_zeroeval(raw_input)
                if processed_input is not None:
                    kwargs["input"] = processed_input

                # Try to get base_url from client instance
                base_url = None
                if args and hasattr(args[0], 'base_url'):
                    base_url = str(args[0].base_url)

                span_attributes = {
                    "service.name": "openai",
                    "provider": get_provider_from_base_url(base_url),
                    "model": kwargs.get("model"),
                    "endpoint": "responses.parse",
                    "base_url": base_url,
                }
                if zeroeval_metadata:
                    span_attributes["variables"] = zeroeval_metadata.get("variables", {})
                    task_id = zeroeval_metadata.get("task")
                    span_attributes["task"] = task_id
                    span_attributes["zeroeval"] = zeroeval_metadata
                    
                    # Attempt to patch model from prompt_version_id if present
                    try:
                        prompt_version_id = zeroeval_metadata.get("prompt_version_id")
                        if prompt_version_id:
                            from ....client import ZeroEval as _PromptClient
                            _pc = _PromptClient()
                            _model = _pc.get_model_for_prompt_version(prompt_version_id=prompt_version_id)
                            if _model:
                                kwargs["model"] = _model
                    except Exception:
                        pass
                    
                    # Log task metadata information
                    self._log_task_metadata(task_id, zeroeval_metadata, "OpenAI responses")

                base_url = None
                if hasattr(responses_instance, "_client") and hasattr(responses_instance._client, "base_url"):
                    try:
                        base_url = str(responses_instance._client.base_url)
                    except Exception:
                        base_url = None
                if base_url:
                    span_attributes["base_url"] = base_url

                if "tools" in kwargs and kwargs["tools"]:
                    tools_info = []
                    for tool in kwargs["tools"]:
                        if isinstance(tool, dict) and tool.get("type") == "function":
                            tools_info.append({
                                "type": "function",
                                "name": tool.get("function", {}).get("name"),
                                "description": tool.get("function", {}).get("description"),
                            })
                    span_attributes["tools"] = tools_info
                    span_attributes["tools_raw"] = kwargs["tools"]

                if "tool_choice" in kwargs:
                    span_attributes["tool_choice"] = kwargs["tool_choice"]

                if "text_format" in kwargs and kwargs["text_format"] is not None:
                    try:
                        # Extract Pydantic schema in OpenAI's response_format structure
                        response_format = extract_pydantic_schema_for_openai(kwargs["text_format"])
                        if response_format:
                            span_attributes["text_format_response_format"] = response_format
                            # Also keep the simple name for backward compatibility
                            span_attributes["text_format"] = response_format["json_schema"]["name"]
                        else:
                            # Fallback to simple name if not a Pydantic model
                            span_attributes["text_format"] = getattr(kwargs["text_format"], "__name__", str(kwargs["text_format"]))
                    except Exception as e:
                        logger.debug(f"Error capturing text_format: {e}")
                        span_attributes["text_format"] = "<unknown>"

                if normalized_messages:
                    span_attributes["messages"] = normalized_messages

                span = self.tracer.start_span(
                    name="openai.responses.parse",
                    kind="llm",
                    attributes=span_attributes,
                    tags={"integration": "openai", "endpoint": "responses.parse"},
                )
                tracer = self.tracer

                original_base_url = None
                original_api_key = None
                try:
                    model_value = kwargs.get("model", "")
                    if isinstance(model_value, str) and model_value.startswith("zeroeval/"):
                        kwargs["model"] = model_value.split("/", 1)[1]
                        ze_base = (os.getenv("ZEROEVAL_BASE_URL") or os.getenv("ZEROEVAL_API_URL") or "https://api.zeroeval.com").rstrip("/") + "/v1"
                        ze_api_key = os.getenv("ZEROEVAL_API_KEY")
                        
                        # Update span attributes to reflect ZeroEval routing
                        span.attributes["base_url"] = ze_base
                        span.attributes["api_version"] = "v1"
                        
                        if hasattr(responses_instance, "_client"):
                            if hasattr(responses_instance._client, "base_url"):
                                original_base_url = responses_instance._client.base_url
                                responses_instance._client.base_url = ze_base
                            if ze_api_key and hasattr(responses_instance._client, "api_key"):
                                original_api_key = responses_instance._client.api_key
                                responses_instance._client.api_key = ze_api_key
                        elif hasattr(responses_instance, "base_url"):
                            original_base_url = responses_instance.base_url
                            responses_instance.base_url = ze_base
                            if ze_api_key and hasattr(responses_instance, "api_key"):
                                original_api_key = responses_instance.api_key
                                responses_instance.api_key = ze_api_key

                    response = original(*args, **kwargs)

                    elapsed = time.time() - start_time

                    usage = getattr(response, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                        span.attributes.update({
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                        })
                        span.attributes["usage"] = {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": getattr(usage, "total_tokens", input_tokens + output_tokens),
                        }

                    output_text = getattr(response, "output_text", "")
                    if hasattr(response, "id"):
                        span.attributes["openai_id"] = response.id
                    if hasattr(response, "system_fingerprint"):
                        span.attributes["system_fingerprint"] = response.system_fingerprint

                    output_items = getattr(response, "output", [])
                    tool_calls = []
                    reasoning_trace = []
                    for item in output_items:
                        if hasattr(item, "type"):
                            if item.type == "function_call":
                                tool_call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {"name": getattr(item, "name", None), "arguments": getattr(item, "arguments", "")},
                                })
                            elif item.type == "reasoning" and hasattr(item, "summary"):
                                for summary in item.summary:
                                    if hasattr(summary, "text"):
                                        reasoning_trace.append(summary.text)

                    if tool_calls:
                        span.attributes["tool_calls"] = tool_calls
                        try:
                            existing_messages = span.attributes.get("messages") or normalized_messages or []
                        except Exception:
                            existing_messages = normalized_messages or []
                        try:
                            messages_with_assistant = list(existing_messages)
                        except Exception:
                            messages_with_assistant = []
                        messages_with_assistant.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                        span.attributes["messages"] = messages_with_assistant
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

                    throughput = (len(output_text) / elapsed) if (output_text and elapsed > 0) else 0
                    span.attributes["throughput"] = round(throughput, 2)

                    combined_output = output_text if output_text else (None if tool_calls else None)
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(normalized_messages)),
                        output_data=combined_output,
                    )

                    tracer.end_span(span)
                    if isinstance(response, dict) and not hasattr(response, "to_dict"):
                        return _ResponseWrapper(response)
                    return response
                except Exception as e:
                    span.set_error(code=type(e).__name__, message=str(e), stack=getattr(e, "__traceback__", None))
                    tracer.end_span(span)
                    raise
                finally:
                    try:
                        if hasattr(responses_instance, "_client"):
                            if original_base_url is not None and hasattr(responses_instance._client, "base_url"):
                                responses_instance._client.base_url = original_base_url
                            if original_api_key is not None and hasattr(responses_instance._client, "api_key"):
                                responses_instance._client.api_key = original_api_key
                        else:
                            if original_base_url is not None and hasattr(responses_instance, "base_url"):
                                responses_instance.base_url = original_base_url
                            if original_api_key is not None and hasattr(responses_instance, "api_key"):
                                responses_instance.api_key = original_api_key
                    except Exception:
                        pass

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
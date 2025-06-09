import json
from functools import wraps
from typing import Any, Callable
from ..base import Integration

class OpenAIIntegration(Integration):
    """Integration for OpenAI's API client."""
    
    PACKAGE_NAME = "openai"

    def _serialize_messages(self, messages: Any) -> Any:
        if not messages:
            return messages
        
        serializable_messages = []
        for message in messages:
            if hasattr(message, "model_dump"):
                # For Pydantic models (openai>=1.0)
                serializable_messages.append(message.model_dump())
            elif isinstance(message, dict):
                serializable_messages.append(message)
            else:
                # Fallback for other types
                try:
                    serializable_messages.append(str(message))
                except:
                    serializable_messages.append("<unserializable_message>")
        return serializable_messages

    def setup(self) -> None:
        try:
            import openai
            # Patch the OpenAI class itself
            self._patch_method(
                openai.OpenAI,
                "__init__",
                self._wrap_init
            )
        except Exception as e:
            print(f"Failed to setup OpenAI integration: {e}")
            pass

    def _wrap_init(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapper(client_instance, *args: Any, **kwargs: Any) -> Any:
            # Call original init first
            result = original(client_instance, *args, **kwargs)
            
            # Then patch the completions.create method
            self._patch_method(
                client_instance.chat.completions,
                "create",
                self._wrap_chat_completion
            )
            
            return result
        return wrapper

    def _wrap_chat_completion(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            start_time = time.time()
            is_streaming = kwargs.get('stream', False)
            
            # Add stream_options for usage stats if streaming
            if is_streaming:
                kwargs['stream_options'] = {"include_usage": True}
            
            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                attributes={
                    "service.name": "openai",
                    "kind": "llm",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": self._serialize_messages(kwargs.get("messages")),
                    "streaming": is_streaming,
                }
            )
            tracer = self.tracer  # capture for inner scope
            
            try:
                response = original(*args, **kwargs)
                
                if is_streaming:
                    # Wrap the streaming response so that it behaves like the original
                    # OpenAI context-manager while still letting us capture metrics.

                    integration_self = self  # capture outer instance for inner class access

                    class _StreamingResponseProxy:
                        """Proxy that mimics OpenAI's streaming response object while capturing metrics."""

                        def __init__(self, _resp):
                            self._resp = _resp
                            self._first_token_time: float | None = None
                            self._full_response: str = ""
                            self._stream_has_finished: bool = False

                        # ------------------------------------------------------------------
                        # Context-manager interface
                        # ------------------------------------------------------------------
                        def __enter__(self):
                            entered = self._resp.__enter__() if hasattr(self._resp, "__enter__") else self._resp
                            self._resp = entered
                            return self

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            if hasattr(self._resp, "__exit__"):
                                self._resp.__exit__(exc_type, exc_val, exc_tb)

                            if exc_type is not None:
                                span.set_error(code=exc_type.__name__, message=str(exc_val), stack=exc_tb)

                            if not self._stream_has_finished:
                                self._finalise_span()

                            # Do *not* suppress exceptions – propagate them.
                            return False

                        # ------------------------------------------------------------------
                        # Iterator interface
                        # ------------------------------------------------------------------
                        def __iter__(self):
                            try:
                                for chunk in self._resp:
                                    # Usage-only chunks (statistics) have no choices
                                    if not getattr(chunk, "choices", None):
                                        usage = getattr(chunk, "usage", None)
                                        if usage:
                                            span.attributes.update(
                                                {
                                                    "inputTokens": usage.prompt_tokens,
                                                    "outputTokens": usage.completion_tokens,
                                                }
                                            )
                                        continue

                                    # Content chunks -----------------------------------------------------
                                    delta = chunk.choices[0].delta
                                    if delta and getattr(delta, "content", None):
                                        if self._first_token_time is None:
                                            self._first_token_time = time.time()
                                            span.attributes["latency"] = round(self._first_token_time - start_time, 4)
                                        self._full_response += delta.content  # type: ignore[attr-defined]
                                    yield chunk
                            except Exception as exc:
                                span.set_error(
                                    code=exc.__class__.__name__,
                                    message=str(exc),
                                    stack=getattr(exc, "__traceback__", None),
                                )
                                raise
                            finally:
                                self._stream_has_finished = True
                                self._finalise_span()

                        # ------------------------------------------------------------------
                        # Attribute delegation – e.g. .get_final_completion()
                        # ------------------------------------------------------------------
                        def __getattr__(self, item):
                            return getattr(self._resp, item)

                        # ------------------------------------------------------------------
                        # Helpers
                        # ------------------------------------------------------------------
                        def _finalise_span(self):
                            if span.end_time is not None:
                                return  # Already closed elsewhere

                            elapsed_time = time.time() - start_time
                            throughput = len(self._full_response) / elapsed_time if elapsed_time > 0 else 0
                            span.attributes.update({"throughput": round(throughput, 2)})
                            span.set_io(
                                input_data=json.dumps(
                                    integration_self._serialize_messages(kwargs.get("messages"))
                                ),
                                output_data=self._full_response,
                            )
                            tracer.end_span(span)

                    return _StreamingResponseProxy(response)
                else:
                    # ---------------- non-streaming ----------------
                    elapsed_time = time.time() - start_time
                    usage = getattr(response, "usage", None)
                    if usage:
                        span.attributes.update(
                            {
                                "inputTokens": usage.prompt_tokens,
                                "outputTokens": usage.completion_tokens,
                            }
                        )

                    output = None
                    if hasattr(response, "choices") and response.choices:
                        message = response.choices[0].message
                        output = message.content if message else None

                    output_length = len(output) if output else 0
                    throughput = output_length / elapsed_time if elapsed_time > 0 else 0
                    span.attributes.update({"throughput": round(throughput, 2)})
                    span.set_io(
                        input_data=json.dumps(self._serialize_messages(kwargs.get("messages"))),
                        output_data=output,
                    )
                    tracer.end_span(span)
                    return response
            except Exception as e:
                # Capture any error that bubbles up before response iteration.
                span.set_error(
                    code=e.__class__.__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None),
                )
                tracer.end_span(span)
                raise
        return wrapper
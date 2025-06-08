from functools import wraps
from typing import Any, Callable
from ..base import Integration

class OpenAIIntegration(Integration):
    """Integration for OpenAI's API client."""
    
    PACKAGE_NAME = "openai"

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
                    "messages": kwargs.get("messages"),
                    "streaming": is_streaming,
                }
            )
            tracer = self.tracer  # capture for inner scope
            
            try:
                response = original(*args, **kwargs)
                
                if is_streaming:
                    # Wrap the streaming response so that it behaves like the original
                    # OpenAI context-manager while still letting us capture metrics.
                    
                    import functools
                    import types

                    class _StreamingResponseProxy:
                        """Proxy that mimics OpenAI's streaming response object.

                        It forwards attribute access to the original *response* object
                        and implements the iterator + context-manager protocols so that
                        callers (e.g. LangChain) can use it transparently:

                            with client.chat.completions.create(..., stream=True) as r:
                                for chunk in r:
                                    ...
                        """

                        def __init__(self, _resp):
                            self._resp = _resp
                            self._first_token_time = None
                            self._full_response = ""
                            self._stream_has_finished = False

                        # ------------------------------------------------------------------
                        # Context-manager interface
                        # ------------------------------------------------------------------
                        def __enter__(self):
                            # The underlying object is also a context-manager.
                            entered = (
                                self._resp.__enter__() if hasattr(self._resp, "__enter__") else self._resp
                            )
                            # Replace the wrapped response with the one returned from __enter__
                            self._resp = entered
                            return self

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            # Propagate to underlying response first so that any clean-up
                            # happens before we finalise the span.
                            if hasattr(self._resp, "__exit__"):
                                self._resp.__exit__(exc_type, exc_val, exc_tb)

                            if exc_type is not None:
                                span.set_error(
                                    code=exc_type.__name__,
                                    message=str(exc_val),
                                    stack=exc_tb,
                                )

                            # Finalise span only once.
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
                                    # Usage chunks have no choices – handle separately.
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

                                    # Only measure latency on the *first* content token.
                                    delta = chunk.choices[0].delta
                                    if delta and getattr(delta, "content", None):
                                        if self._first_token_time is None:
                                            self._first_token_time = time.time()
                                            span.attributes["latency"] = round(
                                                self._first_token_time - start_time, 4
                                            )
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
                                return  # no-op if already closed elsewhere

                            elapsed_time = time.time() - start_time
                            throughput = (
                                len(self._full_response) / elapsed_time if elapsed_time > 0 else 0
                            )
                            span.attributes.update({"throughput": round(throughput, 2)})
                            span.set_io(
                                input_data=str(kwargs.get("messages")),
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
                        input_data=str(kwargs.get("messages")),
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
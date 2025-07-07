import json
from functools import wraps
from typing import Any, Callable
from ..base import Integration
import inspect
import logging
import types

# use the package logger so callers can enable it if they want
logger = logging.getLogger(__name__)

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
            # Patch the *async* client as well
            self._patch_method(
                openai.AsyncOpenAI,
                "__init__",
                self._wrap_init
            )
        except Exception as e:
            print(f"Failed to setup OpenAI integration: {e}")
            pass

    def _wrap_init(self, original: Callable) -> Callable:
        @wraps(original)
        def wrapper(client_instance, *args: Any, **kwargs: Any) -> Any:
            # Call the real __init__
            result = original(client_instance, *args, **kwargs)
            
            # Decide once: are we dealing with the async or sync client?
            import openai
            if isinstance(client_instance, openai.AsyncOpenAI):
                logger.debug("Patching AsyncOpenAI.create with async wrapper")
                patched = self._wrap_chat_completion_async
            else:
                logger.debug("Patching OpenAI.create with sync wrapper")
                patched = self._wrap_chat_completion_sync

            # replace .create
            self._patch_method(client_instance.chat.completions, "create", patched)
            
            return result
        return wrapper

    # ------------------------------------------------------------------+
    #  Async  wrapper –   client = openai.AsyncOpenAI                   |
    # ------------------------------------------------------------------+
    def _wrap_chat_completion_async(self, original: Callable) -> Callable:  # noqa: C901 (length)
        import time, json, inspect

        @wraps(original)
        async def wrapper(*args: Any, **kwargs: Any):  # type: ignore[return-type]
            start_time = time.time()
            is_streaming = kwargs.get("stream", False)

            # openai-native models only
            if is_streaming and (not isinstance(kwargs.get("model"), str) or "/" not in kwargs["model"]):
                kwargs["stream_options"] = {"include_usage": True}

            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                attributes={
                    "service.name": "openai",
                    "kind": "llm",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": self._serialize_messages(kwargs.get("messages")),
                    "streaming": is_streaming,
                },
                tags={"integration": "openai"},
            )
            tracer = self.tracer

            try:
                response = await original(*args, **kwargs)

                # ---------------- STREAMING (ASYNC) -----------------
                if is_streaming:
                    logger.debug("Async wrapper -> returning *async generator* passthrough")

                    async def _relay():  # captures variables from outer scope
                        full_response: str = ""
                        first_token_time: float | None = None

                        try:
                            async for chunk in response:
                                #
                                # 1️⃣  "meta" packets with neither `choices` nor `usage`
                                #     (e.g.   {"event":"message_start", ...} )
                                #
                                if not getattr(chunk, "choices", None) and not getattr(chunk, "usage", None):
                                    # just pass it through – stream is *not* finished yet
                                    yield chunk
                                    continue

                                #
                                # 2️⃣  Usage-only packet – last real packet of the stream
                                #
                                if not getattr(chunk, "choices", None) and getattr(chunk, "usage", None):
                                    usage = chunk.usage            # type: ignore[attr-defined]
                                    span.attributes.update(
                                        {
                                            "inputTokens": usage.prompt_tokens,
                                            "outputTokens": usage.completion_tokens,
                                        }
                                    )
                                    yield chunk
                                    # do *not* continue; we want to stay in the loop until
                                    # the server closes the connection so that we fire the
                                    # finally-clause below.
                                    continue

                                # Content chunk
                                delta = chunk.choices[0].delta
                                if delta and getattr(delta, "content", None):
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                        span.attributes["latency"] = round(
                                            first_token_time - start_time, 4
                                        )
                                    full_response += delta.content
                                yield chunk
                        except Exception as exc:
                            span.set_error(
                                code=exc.__class__.__name__,
                                message=str(exc),
                                stack=getattr(exc, "__traceback__", None),
                            )
                            raise
                        finally:
                            # Finalise span once the async generator is exhausted
                            elapsed = time.time() - start_time
                            throughput = len(full_response) / elapsed if elapsed > 0 else 0
                            span.attributes.update({"throughput": round(throughput, 2)})
                            span.set_io(
                                input_data=json.dumps(
                                    self._serialize_messages(kwargs.get("messages"))
                                ),
                                output_data=full_response,
                            )
                            tracer.end_span(span)

                    return _relay()  # returns an *awaitable* async generator

                # ---------- non-streaming ----------
                elapsed = time.time() - start_time
                usage = getattr(response, "usage", None)
                if usage:
                    span.attributes.update(
                        {"inputTokens": usage.prompt_tokens, "outputTokens": usage.completion_tokens}
                    )
                message = response.choices[0].message if response.choices else None
                output = message.content if message else None
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
    def _wrap_chat_completion_sync(self, original: Callable) -> Callable:  # noqa: C901 (length)
        import time, json

        @wraps(original)
        def wrapper(*args: Any, **kwargs: Any):
            start_time = time.time()
            is_streaming = kwargs.get("stream", False)

            if is_streaming and (not isinstance(kwargs.get("model"), str) or "/" not in kwargs["model"]):
                kwargs["stream_options"] = {"include_usage": True}

            span = self.tracer.start_span(
                name="openai.chat.completions.create",
                attributes={
                    "service.name": "openai",
                    "kind": "llm",
                    "provider": "openai",
                    "model": kwargs.get("model"),
                    "messages": self._serialize_messages(kwargs.get("messages")),
                    "streaming": is_streaming,
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
                message = response.choices[0].message if response.choices else None
                output = message.content if message else None
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

class _StreamingResponseProxy:
    """Proxy that mimics OpenAI's streaming response object while capturing metrics for both sync and async streams."""

    def __init__(self, _resp, span, tracer, integration_instance, request_kwargs, start_time, is_async=False):
        self._resp = _resp
        self.span = span
        self.tracer = tracer
        self.integration_instance = integration_instance
        self.request_kwargs = request_kwargs
        self.start_time = start_time
        self._is_async = is_async
        self._first_token_time: float | None = None
        self._full_response: str = ""
        self._stream_has_finished: bool = False

    def __enter__(self):
        entered = self._resp.__enter__() if hasattr(self._resp, "__enter__") else self._resp
        self._resp = entered
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._resp, "__exit__"):
            self._resp.__exit__(exc_type, exc_val, exc_tb)

        if exc_type is not None:
            self.span.set_error(code=exc_type.__name__, message=str(exc_val), stack=exc_tb)

        if not self._stream_has_finished:
            self._finalise_span()
        return False

    async def __aenter__(self):
        if hasattr(self._resp, "__aenter__"):
            self._resp = await self._resp.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._resp, "__aexit__"):
            await self._resp.__aexit__(exc_type, exc_val, exc_tb)

        if exc_type is not None:
            self.span.set_error(code=exc_type.__name__, message=str(exc_val), stack=exc_tb)

        if not self._stream_has_finished:
            self._finalise_span()
        return False

    def _process_chunk(self, chunk):
        import time
        if not getattr(chunk, "choices", None):
            usage = getattr(chunk, "usage", None)
            if usage:
                self.span.attributes.update(
                    {
                        "inputTokens": usage.prompt_tokens,
                        "outputTokens": usage.completion_tokens,
                    }
                )
            return

        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            if self._first_token_time is None:
                self._first_token_time = time.time()
                self.span.attributes["latency"] = round(self._first_token_time - self.start_time, 4)
            self._full_response += delta.content

    # ------------------------------------------------------------------
    # Iterator interfaces (sync *and* async)
    # ------------------------------------------------------------------

    # -- synchronous ----------------------------------------------------
    def __iter__(self):
        if self._is_async:
            raise TypeError("Use 'async for' with an async stream.")
        return self

    def __next__(self):
        try:
            chunk = next(self._resp)
        except StopIteration:
            self._stream_has_finished = True
            self._finalise_span()
            raise
        self._process_chunk(chunk)
        return chunk

    # -- asynchronous ---------------------------------------------------
    def __aiter__(self):
        if not self._is_async:
            # allow 'async for' on sync streams too
            return self  # __anext__ will delegate to __next__
        return self

    async def __anext__(self):
        if not self._is_async:
            # delegate to sync __next__ in a thread
            import asyncio, functools
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(self.__next__))

        try:
            chunk = await self._resp.__anext__()  # type: ignore[attr-defined]
        except StopAsyncIteration:
            self._stream_has_finished = True
            self._finalise_span()
            raise
        self._process_chunk(chunk)
        return chunk

    def __getattr__(self, item):
        return getattr(self._resp, item)

    def _finalise_span(self):
        import time
        if self.span.end_time is not None:
            return

        elapsed_time = time.time() - self.start_time
        throughput = len(self._full_response) / elapsed_time if elapsed_time > 0 else 0
        self.span.attributes.update({"throughput": round(throughput, 2)})
        self.span.set_io(
            input_data=json.dumps(self.integration_instance._serialize_messages(self.request_kwargs.get("messages"))),
            output_data=self._full_response,
        )
        self.tracer.end_span(self.span)
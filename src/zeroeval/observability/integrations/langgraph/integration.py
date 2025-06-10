from ..base import Integration

import inspect
import time
import json
from typing import Any, Callable


class LangGraphIntegration(Integration):
    """ZeroEval integration for LangGraph graphs (StateGraph / Graph).

    The integration instruments compiled graphs (``CompiledGraph`` and
    ``CompiledStateGraph`` instances) produced by ``graph.compile()`` so that
    each invocation (``invoke``/``ainvoke``) and streaming run
    (``stream``/``astream``) is wrapped in a ZeroEval span.  Individual LangChain
    Runnables executed *inside* the graph are already covered by the existing
    LangChain integration – this class focuses on adding a *parent* span that
    represents the end-to-end graph execution.
    """

    PACKAGE_NAME = "langgraph"

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def setup(self) -> None:  # noqa: D401
        """Patch Graph.compile / StateGraph.compile so we can instrument the
        returned compiled graph instances.
        """
        try:
            from langgraph.graph.graph import Graph  # type: ignore
            from langgraph.graph.state import StateGraph  # type: ignore

            # Patch *both* compile methods (Graph + StateGraph inherit separate
            # compile implementations).
            self._patch_method(Graph, "compile", self._wrap_compile)
            self._patch_method(StateGraph, "compile", self._wrap_compile)
        except Exception as exc:  # pragma: no cover – optional dependency
            print(f"[ZeroEval] Failed to setup LangGraph integration: {exc}")

    # ------------------------------------------------------------------
    # Internal – compile wrapper
    # ------------------------------------------------------------------
    def _wrap_compile(self, original: Callable) -> Callable:
        """Return a wrapper around *.compile()* which instruments the returned
        compiled graph instance (per-instance patch).
        """

        def wrapper(graph_self, *args: Any, **kwargs: Any):  # type: ignore
            compiled_graph = original(graph_self, *args, **kwargs)

            # Instrument the resulting compiled graph *instance* so that we
            # create a parent span covering the entire run.
            for method_name in ("invoke", "ainvoke", "stream", "astream"):
                if hasattr(compiled_graph, method_name):
                    try:
                        self._patch_method(
                            compiled_graph,  # instance target
                            method_name,
                            self._build_graph_wrapper,
                        )
                    except Exception:
                        # Best-effort – skip if already patched at instance level.
                        pass

            return compiled_graph

        return wrapper

    # ------------------------------------------------------------------
    # Wrapper factory for compiled graph methods
    # ------------------------------------------------------------------
    def _build_graph_wrapper(self, original: Callable) -> Callable:  # noqa: C901
        """Generate a patched version of *original* that records a span.

        Supports sync + async callables (invoke / stream / ainvoke / astream).
        """
        is_async = inspect.iscoroutinefunction(original)
        method_name = original.__name__

        # ------------------------------------------------------------------
        # Async version
        # ------------------------------------------------------------------
        if is_async:

            async def async_wrapper(compiled_self, *args: Any, **kwargs: Any):  # type: ignore  # noqa: ANN001
                start_time = time.time()
                span = self._start_span(compiled_self, method_name)
                try:
                    result = await original(compiled_self, *args, **kwargs)

                    if method_name == "astream":
                        return self._wrap_async_generator(result, span, start_time)

                    # Non-stream async call (ainvoke)
                    self._finalise_span(span, start_time, args, kwargs, result)
                    return result
                except Exception as exc:
                    self._record_error(span, exc)
                    raise

            return async_wrapper

        # ------------------------------------------------------------------
        # Sync version
        # ------------------------------------------------------------------
        def sync_wrapper(compiled_self, *args: Any, **kwargs: Any):  # type: ignore  # noqa: ANN001
            start_time = time.time()
            span = self._start_span(compiled_self, method_name)
            try:
                result = original(compiled_self, *args, **kwargs)

                if method_name == "stream":
                    return self._wrap_sync_generator(result, span, start_time)

                # Non-stream call (invoke)
                self._finalise_span(span, start_time, args, kwargs, result)
                return result
            except Exception as exc:
                self._record_error(span, exc)
                raise

        return sync_wrapper

    # ------------------------------------------------------------------
    # Span helpers
    # ------------------------------------------------------------------
    def _start_span(self, compiled_self, method_name: str):  # noqa: ANN001
        attributes = {
            "service.name": "langgraph",
            "kind": "graph",
            "class": compiled_self.__class__.__name__,
            "method": method_name,
        }

        # Graph instances may have a .name attribute or .id – capture if present
        for attr in ("name", "id"):
            if hasattr(compiled_self, attr):
                attributes[attr] = getattr(compiled_self, attr)
                break

        return self.tracer.start_span(name=f"langgraph.{method_name}", attributes=attributes)

    def _finalise_span(self, span, start_time: float, args, kwargs, output):  # noqa: ANN001
        elapsed = time.time() - start_time
        span.attributes["latency"] = round(elapsed, 4)

        try:
            span.set_io(input_data=str(args or kwargs), output_data=str(output))
        except Exception:
            span.set_io(input_data="<unserialisable>", output_data="<unserialisable>")

        self.tracer.end_span(span)

    def _record_error(self, span, exc: Exception):  # noqa: ANN001
        span.set_error(
            code=exc.__class__.__name__,
            message=str(exc),
            stack=getattr(exc, "__traceback__", None),
        )
        self.tracer.end_span(span)

    # ------------------------------------------------------------------
    # Generator wrappers
    # ------------------------------------------------------------------
    def _wrap_sync_generator(self, gen, span, start_time: float):  # noqa: ANN001
        first_chunk_time = None

        def _generator():
            nonlocal first_chunk_time
            try:
                for chunk in gen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["latency"] = round(first_chunk_time - start_time, 4)
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                self._finalise_span(span, start_time, None, None, "<stream>")

        return _generator()

    def _wrap_async_generator(self, agen, span, start_time: float):  # noqa: ANN001
        first_chunk_time = None

        async def _async_generator():
            nonlocal first_chunk_time
            try:
                async for chunk in agen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["latency"] = round(first_chunk_time - start_time, 4)
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                self._finalise_span(span, start_time, None, None, "<stream>")

        return _async_generator() 
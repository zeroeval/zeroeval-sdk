import inspect
import time
import types
from typing import Any, Callable

from ..base import Integration


class LangChainIntegration(Integration):
    """Integration for LangChain's Runnable & LLM execution pipeline.

    This patches the six canonical entry-points on ``langchain_core.runnables.base.Runnable`` to
    transparently create ZeroEval spans for every synchronous / asynchronous execution path:

    • invoke / ainvoke  – single-input execution
    • stream / astream  – streaming generators (sync & async)
    • batch  / abatch   – batch execution

    The wrapper is deliberately generic: it does **not** depend on concrete chain / model types
    and therefore covers LLMs, Chains, Agents, Tools, custom Runnables – anything built on
    top of LangChain's Runnable abstraction.
    """

    # Accept either the monolithic 'langchain' package (pre-0.1) or the
    # lean runtime split introduced in 0.1+: 'langchain-core'.
    PACKAGE_NAME = "langchain_core"

    @classmethod
    def is_available(cls) -> bool:  # noqa: D401
        """Return True if *either* langchain-core or langchain is importable."""
        import importlib

        for mod in ("langchain_core", "langchain"):
            try:
                importlib.import_module(mod)
                return True
            except ImportError:
                continue
        return False

    # ---------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------
    def setup(self) -> None:
        """Patch Runnable methods once LangChain is importable."""
        from langchain_core.runnables.base import (
            Runnable,  # pylint: disable=import-error
        )

        # Gather the base class and *all* subclasses so that overrides on custom
        # Runnables (e.g. RunnableSequence) are instrumented as well.
        def _iter_all_subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from _iter_all_subclasses(sub)

        runnable_classes = [Runnable, *_iter_all_subclasses(Runnable)]

        # ------------------------------------------------------------------
        # Exhaustive instrumentation for Runnable entrypoints
        # ------------------------------------------------------------------
        runnable_method_names = (
            "invoke",
            "stream",
            "batch",
            "ainvoke",
            "astream",
            "abatch",
        )

        for method_name in runnable_method_names:
            for cls in runnable_classes:
                # Only patch if the attribute exists on the target class.
                if hasattr(cls, method_name):
                    try:
                        self._patch_method(cls, method_name, self._build_runnable_wrapper)
                    except Exception as exc:  # pragma: no cover – best-effort
                        print(
                            f"[ZeroEval] Failed patching {cls.__name__}.{method_name}: {exc}"
                        )

        # ------------------------------------------------------------------
        # Patch *future* subclasses created after this point by hooking
        # Runnable.__init_subclass__. This ensures we don\'t miss classes that
        # are imported later (e.g. Prompt templates, Tool wrappers).
        # ------------------------------------------------------------------
        original_init_subclass = Runnable.__init_subclass__

        integration_self = self  # capture for closure

        @classmethod  # type: ignore[misc]
        def _ze_init_subclass(cls, **kwargs):  # noqa: D401, ANN001  pylint: disable=unused-argument
            # Call LangChain\'s original __init_subclass__ first
            try:
                # Handle both regular methods and builtin methods
                if hasattr(original_init_subclass, '__get__'):
                    # It's a regular method, use descriptor protocol
                    bound_init = original_init_subclass.__get__(cls, cls)
                    bound_init(**kwargs)  # type: ignore[misc]
                else:
                    # It's a builtin method, call directly
                    original_init_subclass(**kwargs)
            except TypeError:
                # Some versions accept no kwargs at all.
                try:
                    if hasattr(original_init_subclass, '__get__'):
                        bound_init = original_init_subclass.__get__(cls, cls)
                        bound_init()
                    else:
                        original_init_subclass()
                except Exception:
                    # Fallback: just ignore if we can't call it
                    pass

            # Patch the new subclass\'s methods
            for method_name in (
                "invoke",
                "stream",
                "batch",
                "ainvoke",
                "astream",
                "abatch",
            ):
                # Only patch if the attribute exists on the target class.
                if hasattr(cls, method_name):
                    try:
                        integration_self._patch_method(
                            cls, method_name, integration_self._build_runnable_wrapper
                        )
                    except Exception:
                        # Ignore – best effort instrumentation
                        pass

        # Avoid double-hooking if someone else already replaced it
        if getattr(Runnable.__init_subclass__, "__ze_patched__", False) is False:
            _ze_init_subclass.__ze_patched__ = True
            Runnable.__init_subclass__ = _ze_init_subclass

        # ------------------------------------------------------------------
        # Instrument additional LangChain abstractions (Tools, LLMs, Retrievers)
        # ------------------------------------------------------------------

        try:
            from langchain_core.tools import BaseTool  # pylint: disable=import-error

            self._instrument_class_methods(
                BaseTool,
                (
                    "run",
                    "arun",
                ),
            )
        except Exception:  # pragma: no cover – missing module / version mismatch
            pass

        try:
            from langchain_core.language_models.base import (
                BaseLanguageModel,  # pylint: disable=import-error
            )

            self._instrument_class_methods(
                BaseLanguageModel,
                (
                    "generate",
                    "agenerate",
                ),
            )
        except Exception:  # pragma: no cover
            pass

        try:
            from langchain_core.retrievers import (
                BaseRetriever,  # pylint: disable=import-error
            )

            self._instrument_class_methods(
                BaseRetriever,
                (
                    "get_relevant_documents",
                    "aget_relevant_documents",
                    "__call__",
                ),
            )
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Wrapper factory
    # ------------------------------------------------------------------
    def _build_runnable_wrapper(self, original: Callable) -> Callable:
        """Return a patched function matching *original* (sync or async)."""

        is_async = inspect.iscoroutinefunction(original)

        if is_async:

            async def async_wrapper(runnable_self, *args: Any, **kwargs: Any):  # type: ignore
                span = self._start_runnable_span(runnable_self, original.__name__, kwargs)
                start_time = time.time()
                try:
                    result = await original(runnable_self, *args, **kwargs)

                    # Handle async generators (e.g. astream)
                    if inspect.isasyncgen(result):
                        return self._wrap_async_generator(result, span, start_time)

                    # Non-streaming async result
                    self._finalise_span(span, start_time, args, kwargs, result)
                    return result
                except Exception as exc:
                    self._record_error(span, exc)
                    raise

            return async_wrapper

        # ------------------------------------------------------------------
        # synchronous path
        # ------------------------------------------------------------------
        def sync_wrapper(runnable_self, *args: Any, **kwargs: Any):  # type: ignore
            span = self._start_runnable_span(runnable_self, original.__name__, kwargs)
            start_time = time.time()
            try:
                result = original(runnable_self, *args, **kwargs)

                # Handle sync generators (e.g. stream)
                if inspect.isgenerator(result):
                    return self._wrap_sync_generator(result, span, start_time)

                # Non-streaming sync result
                self._finalise_span(span, start_time, args, kwargs, result)
                return result
            except Exception as exc:
                self._record_error(span, exc)
                raise

        return sync_wrapper

    # ------------------------------------------------------------------
    # Generator wrappers
    # ------------------------------------------------------------------
    def _wrap_sync_generator(self, gen: types.GeneratorType, span, start_time: float):  # noqa: ANN001
        """Yield from *gen* while updating *span* metrics."""
        first_chunk_time = None
        collected_output = []

        def _wrapper():
            nonlocal first_chunk_time
            try:
                for chunk in gen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["latency"] = round(first_chunk_time - start_time, 4)
                    collected_output.append(chunk)
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                self._finalise_span(span, start_time, None, None, collected_output)

        return _wrapper()

    def _wrap_async_generator(self, agen: types.AsyncGeneratorType, span, start_time: float):  # noqa: ANN001
        """Yield from *agen* while updating *span* metrics (async)."""
        first_chunk_time = None
        collected_output = []

        async def _wrapper():
            nonlocal first_chunk_time
            try:
                async for chunk in agen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["latency"] = round(first_chunk_time - start_time, 4)
                    collected_output.append(chunk)
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                self._finalise_span(span, start_time, None, None, collected_output)

        return _wrapper()

    # ------------------------------------------------------------------
    # Span helpers
    # ------------------------------------------------------------------
    def _start_runnable_span(self, runnable_self, method_name: str, kwargs: dict):  # noqa: ANN001
        """Create + return a Span for a Runnable invocation."""
        # Determine a suitable kind based on the LangChain abstraction
        kind: str = "runnable"
        try:
            from langchain_core.language_models.base import (
                BaseLanguageModel,  # pylint: disable=import-error
            )
            from langchain_core.retrievers import (
                BaseRetriever,  # pylint: disable=import-error
            )
            from langchain_core.tools import BaseTool  # pylint: disable=import-error

            if isinstance(runnable_self, BaseTool):
                kind = "tool"
            elif isinstance(runnable_self, BaseLanguageModel):
                kind = "llm"
            elif isinstance(runnable_self, BaseRetriever):
                kind = "retriever"
        except Exception:  # pragma: no cover – optional deps may be missing
            pass

        attributes = {
            "service.name": "langchain",
            "kind": kind,
            "class": runnable_self.__class__.__name__,
            "method": method_name,
        }
        # Surface model name if available (ChatOpenAI / BaseLanguageModel)
        model_name = getattr(runnable_self, "model_name", None)
        if model_name:
            attributes["model"] = model_name

        # Include run-specific attributes for debugging (keep small!)
        if kwargs:
            for k in ("stream", "temperature", "top_p"):
                if k in kwargs:
                    attributes[k] = kwargs[k]

        return self.tracer.start_span(name=f"langchain.{method_name}", attributes=attributes, tags={"integration": "langchain"})

    def _finalise_span(self, span, start_time: float, args, kwargs, output):  # noqa: ANN001
        """Attach IO + latency + throughput then close span."""
        elapsed = time.time() - start_time
        span.attributes["latency"] = round(elapsed, 4)

        # Only serialise small representations for readability
        try:
            span.set_io(input_data=str(args or kwargs), output_data=str(output))
        except Exception:  # pragma: no cover – defensive
            span.set_io(input_data="<unserialisable>", output_data="<unserialisable>")

        self.tracer.end_span(span)

    def _record_error(self, span, exc: Exception):  # noqa: ANN001
        """Populate error fields on *span*."""
        span.set_error(
            code=exc.__class__.__name__,
            message=str(exc),
            stack=getattr(exc, "__traceback__", None),
        )

    # ------------------------------------------------------------------
    # Generic helper for patching class hierarchies
    # ------------------------------------------------------------------
    def _instrument_class_methods(self, base_cls: type, method_names: tuple[str, ...]):
        """Patch *method_names* on *base_cls* and all its current + future subclasses."""

        # Gather existing subclasses recursively
        def _iter_all_subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from _iter_all_subclasses(sub)

        target_classes = [base_cls, *_iter_all_subclasses(base_cls)]

        for method_name in method_names:
            for cls in target_classes:
                # Only patch if the attribute exists on the target class.
                if hasattr(cls, method_name):
                    try:
                        self._patch_method(cls, method_name, self._build_runnable_wrapper)
                    except Exception as exc:  # pragma: no cover – best-effort
                        print(
                            f"[ZeroEval] Failed patching {cls.__name__}.{method_name}: {exc}"
                        )

        # ------------------------------------------------------------------
        # Ensure *future* subclasses are instrumented via __init_subclass__
        # ------------------------------------------------------------------
        original_init_subclass = base_cls.__init_subclass__

        integration_self = self  # capture for closure

        @classmethod  # type: ignore[misc]
        def _ze_init_subclass(cls, **kwargs):  # noqa: D401, ANN001
            # Call the original hook first
            try:
                # Handle both regular methods and builtin methods
                if hasattr(original_init_subclass, '__get__'):
                    # It's a regular method, use descriptor protocol
                    bound_init = original_init_subclass.__get__(cls, cls)
                    bound_init(**kwargs)  # type: ignore[misc]
                else:
                    # It's a builtin method, call directly
                    original_init_subclass(**kwargs)
            except TypeError:
                # Some versions accept no kwargs at all.
                try:
                    if hasattr(original_init_subclass, '__get__'):
                        bound_init = original_init_subclass.__get__(cls, cls)
                        bound_init()
                    else:
                        original_init_subclass()
                except Exception:
                    # Fallback: just ignore if we can't call it
                    pass

            # Patch the new subclass's methods
            for method in method_names:
                # Only patch if the attribute exists on the target class.
                if hasattr(cls, method):
                    try:
                        integration_self._patch_method(
                            cls, method, integration_self._build_runnable_wrapper
                        )
                    except Exception:
                        pass  # silently ignore – best-effort

        if getattr(base_cls.__init_subclass__, "__ze_patched__", False) is False:
            _ze_init_subclass.__ze_patched__ = True
            base_cls.__init_subclass__ = _ze_init_subclass 
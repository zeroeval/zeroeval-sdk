import inspect
import logging
import threading
import time
from functools import wraps
from typing import Any, Callable

from ..base import Integration


class LangGraphIntegration(Integration):
    """ZeroEval integration for LangGraph graphs (StateGraph / Graph).

    The integration instruments compiled graphs (``CompiledGraph`` and
    ``CompiledStateGraph`` instances) produced by ``graph.compile()`` so that
    each invocation (``invoke``/``ainvoke``) and streaming run
    (``stream``/``astream``) is wrapped in a ZeroEval span.  
    
    Additionally, this integration traces:
    - Individual node executions
    - Edge transitions between nodes
    - State updates and transformations
    - Conditional edge evaluations
    - Checkpointing operations (if enabled)
    - Tool calls within nodes
    """

    PACKAGE_NAME = "langgraph"

    def __init__(self, tracer):
        super().__init__(tracer)
        # Thread-local storage for tracking current graph execution context
        self._thread_local = threading.local()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def setup(self) -> None:  # noqa: D401
        """Patch Graph.compile / StateGraph.compile so we can instrument the
        returned compiled graph instances. Also patch node execution and edges.
        """
        logger = logging.getLogger(__name__)
        
        from langgraph.graph.graph import Graph  # type: ignore
        from langgraph.graph.state import StateGraph  # type: ignore
        
        logger.debug("[LangGraph] Successfully imported Graph and StateGraph")
        
        # Debug: Check what we're about to patch
        logger.debug(f"[LangGraph] Graph.compile exists: {hasattr(Graph, 'compile')}")
        logger.debug(f"[LangGraph] StateGraph.compile exists: {hasattr(StateGraph, 'compile')}")
        
        # Patch *both* compile methods (Graph + StateGraph inherit separate
        # compile implementations).
        try:
            self._patch_method(Graph, "compile", self._wrap_compile)
            logger.debug("[LangGraph] Successfully patched Graph.compile")
        except Exception as e:
            logger.error(f"[LangGraph] Failed to patch Graph.compile: {e}")
            
        try:
            self._patch_method(StateGraph, "compile", self._wrap_compile)
            logger.debug("[LangGraph] Successfully patched StateGraph.compile")
        except Exception as e:
            logger.error(f"[LangGraph] Failed to patch StateGraph.compile: {e}")
        
        # Try to patch node execution if available
        try:
            from langgraph.pregel import Pregel  # type: ignore
            logger.debug("[LangGraph] Pregel module imported")
            
            # List all Pregel methods for debugging
            pregel_methods = [attr for attr in dir(Pregel) if not attr.startswith('__')]
            logger.debug(f"[LangGraph] Pregel methods available: {pregel_methods}")
            
            # Patch the internal node execution method
            if hasattr(Pregel, "_run_node"):
                self._patch_method(Pregel, "_run_node", self._wrap_node_execution)
                logger.debug("[LangGraph] Patched Pregel._run_node")
            else:
                logger.debug("[LangGraph] Pregel._run_node not found")
                
            if hasattr(Pregel, "_arun_node"):
                self._patch_method(Pregel, "_arun_node", self._wrap_node_execution)
                logger.debug("[LangGraph] Patched Pregel._arun_node")
            else:
                logger.debug("[LangGraph] Pregel._arun_node not found")
                
            # Try other potential node execution methods
            for method_name in ["run", "arun", "_exec", "_aexec", "execute", "aexecute"]:
                if hasattr(Pregel, method_name):
                    logger.debug(f"[LangGraph] Found Pregel.{method_name} - considering for patching")
                    
        except ImportError as e:
            logger.warning(f"[LangGraph] Could not import Pregel: {e}")
            pass  # Older versions might not have these internals
            
        # Try to patch checkpointing if available
        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver  # type: ignore
            if hasattr(BaseCheckpointSaver, "put"):
                self._patch_method(BaseCheckpointSaver, "put", self._wrap_checkpoint_put)
                logger.debug("[LangGraph] Patched BaseCheckpointSaver.put")
            if hasattr(BaseCheckpointSaver, "get"):
                self._patch_method(BaseCheckpointSaver, "get", self._wrap_checkpoint_get)
                logger.debug("[LangGraph] Patched BaseCheckpointSaver.get")
        except ImportError:
            logger.warning("[LangGraph] Could not import checkpoint module")
            pass  # Checkpointing might not be used

    # ------------------------------------------------------------------
    # Internal – compile wrapper
    # ------------------------------------------------------------------
    def _wrap_compile(self, original: Callable) -> Callable:
        """Return a wrapper around *.compile()* which instruments the returned
        compiled graph instance (per-instance patch).
        """
        logger = logging.getLogger(__name__)

        def wrapper(graph_self, *args: Any, **kwargs: Any):  # type: ignore
            compiled_graph = original(graph_self, *args, **kwargs)

            # Extract and store graph metadata for better tracing
            try:
                graph_metadata = self._extract_graph_metadata(graph_self, compiled_graph)
                # Store metadata on the compiled graph instance
                compiled_graph._ze_graph_metadata = graph_metadata
            except Exception:
                # Best effort - continue even if metadata extraction fails
                pass

            # IMPORTANT: We need to patch AFTER the object is created because
            # LangChain integration might also patch these methods.
            # By patching at the instance level, we override class-level patches.
            
            # Get the actual methods (which might already be wrapped by LangChain)
            for method_name in ("invoke", "ainvoke", "stream", "astream"):
                if hasattr(compiled_graph, method_name):
                    try:
                        # Get the method (might be wrapped by LangChain already)
                        method = getattr(compiled_graph, method_name)
                        
                        # If it's already wrapped by LangChain, we need to get the original
                        # and wrap it with our wrapper that creates langgraph.* spans
                        if hasattr(method, '__wrapped__'):
                            # It's been wrapped, get the original
                            original_method = method.__wrapped__
                        elif hasattr(method, '__func__'):
                            # It's a bound method, get the function
                            original_method = method.__func__
                        else:
                            # Use it as-is
                            original_method = method
                        
                        # Create our wrapper with the correct method name
                        wrapper_func = self._build_graph_wrapper(original_method, method_name)
                        
                        # Apply it to the instance
                        if hasattr(method, '__self__'):
                            # It's a bound method, rebind with our wrapper
                            import types
                            setattr(compiled_graph, method_name, types.MethodType(wrapper_func, compiled_graph))
                        else:
                            # Just set it
                            setattr(compiled_graph, method_name, wrapper_func)
                            
                        logger.debug(f"[LangGraph] Successfully patched {method_name} on compiled graph instance")
                    except Exception as e:
                        logger.error(f"[LangGraph] Failed to patch {method_name} on instance: {e}")
                        # Best-effort – skip if already patched at instance level.
                        pass

            return compiled_graph

        return wrapper

    # ------------------------------------------------------------------
    # Graph metadata extraction
    # ------------------------------------------------------------------
    def _extract_graph_metadata(self, graph_self, compiled_graph) -> dict[str, Any]:
        """Extract metadata about the graph structure for better tracing."""
        metadata = {
            "nodes": [],
            "edges": [],
            "conditional_edges": [],
            "entry_point": None,
        }
        
        try:
            # Extract nodes
            if hasattr(graph_self, "nodes"):
                metadata["nodes"] = list(graph_self.nodes.keys())
            elif hasattr(graph_self, "_nodes"):
                metadata["nodes"] = list(graph_self._nodes.keys())
                
            # Extract edges
            if hasattr(graph_self, "edges"):
                for source, targets in graph_self.edges.items():
                    if isinstance(targets, (list, tuple)):
                        for target in targets:
                            metadata["edges"].append({"from": source, "to": target})
                    else:
                        metadata["edges"].append({"from": source, "to": targets})
            elif hasattr(graph_self, "_edges"):
                # Handle different edge storage formats
                for edge_info in graph_self._edges:
                    if isinstance(edge_info, tuple) and len(edge_info) >= 2:
                        metadata["edges"].append({"from": edge_info[0], "to": edge_info[1]})
                        
            # Extract conditional edges
            if hasattr(graph_self, "conditional_edges"):
                for source, condition_info in graph_self.conditional_edges.items():
                    metadata["conditional_edges"].append({
                        "from": source,
                        "condition": str(condition_info) if condition_info else "unknown"
                    })
                    
            # Extract entry point
            if hasattr(graph_self, "entry_point"):
                metadata["entry_point"] = graph_self.entry_point
            elif hasattr(compiled_graph, "input"):
                metadata["entry_point"] = str(compiled_graph.input)
                
        except Exception:
            # Best effort extraction
            pass
            
        return metadata

    # ------------------------------------------------------------------
    # Node execution wrapper
    # ------------------------------------------------------------------
    def _wrap_node_execution(self, original: Callable) -> Callable:
        """Wrap individual node execution to create spans for each node."""
        is_async = inspect.iscoroutinefunction(original)
        
        if is_async:
            async def async_wrapper(pregel_self, node_name: str, *args: Any, **kwargs: Any):  # type: ignore
                # Get parent graph span if available
                parent_span = getattr(self._thread_local, 'current_graph_span', None)
                
                span = self.tracer.start_span(
                    name=f"langgraph.node.{node_name}",
                    attributes={
                        "service.name": "langgraph",
                        "kind": "node",
                        "node_name": node_name,
                        "graph_id": getattr(pregel_self, "graph_id", None),
                    },
                    tags={"integration": "langgraph"}
                )
                
                if parent_span:
                    span.parent_id = parent_span.span_id
                    span.trace_id = parent_span.trace_id
                
                start_time = time.time()
                try:
                    result = await original(pregel_self, node_name, *args, **kwargs)
                    
                    # Try to capture state changes
                    try:
                        if len(args) > 0:
                            input_state = str(args[0])[:500]  # Limit size
                            span.set_io(input_data=input_state, output_data=str(result)[:500])
                    except Exception:
                        pass
                        
                    elapsed = time.time() - start_time
                    span.attributes["latency"] = round(elapsed, 4)
                    self.tracer.end_span(span)
                    return result
                except Exception as exc:
                    span.set_error(
                        code=exc.__class__.__name__,
                        message=str(exc),
                        stack=getattr(exc, "__traceback__", None),
                    )
                    self.tracer.end_span(span)
                    raise
                    
            return async_wrapper
        else:
            def sync_wrapper(pregel_self, node_name: str, *args: Any, **kwargs: Any):  # type: ignore
                # Get parent graph span if available
                parent_span = getattr(self._thread_local, 'current_graph_span', None)
                
                span = self.tracer.start_span(
                    name=f"langgraph.node.{node_name}",
                    attributes={
                        "service.name": "langgraph",
                        "kind": "node", 
                        "node_name": node_name,
                        "graph_id": getattr(pregel_self, "graph_id", None),
                    },
                    tags={"integration": "langgraph"}
                )
                
                if parent_span:
                    span.parent_id = parent_span.span_id
                    span.trace_id = parent_span.trace_id
                
                start_time = time.time()
                try:
                    result = original(pregel_self, node_name, *args, **kwargs)
                    
                    # Try to capture state changes
                    try:
                        if len(args) > 0:
                            input_state = str(args[0])[:500]  # Limit size
                            span.set_io(input_data=input_state, output_data=str(result)[:500])
                    except Exception:
                        pass
                        
                    elapsed = time.time() - start_time
                    span.attributes["latency"] = round(elapsed, 4)
                    self.tracer.end_span(span)
                    return result
                except Exception as exc:
                    span.set_error(
                        code=exc.__class__.__name__,
                        message=str(exc),
                        stack=getattr(exc, "__traceback__", None),
                    )
                    self.tracer.end_span(span)
                    raise
                    
            return sync_wrapper

    # ------------------------------------------------------------------
    # Checkpoint wrappers
    # ------------------------------------------------------------------
    def _wrap_checkpoint_put(self, original: Callable) -> Callable:
        """Wrap checkpoint save operations."""
        @wraps(original)
        def wrapper(checkpoint_self, *args: Any, **kwargs: Any):  # type: ignore
            span = self.tracer.start_span(
                name="langgraph.checkpoint.put",
                attributes={
                    "service.name": "langgraph",
                    "kind": "checkpoint",
                    "operation": "save",
                },
                tags={"integration": "langgraph"}
            )
            
            start_time = time.time()
            try:
                result = original(checkpoint_self, *args, **kwargs)
                elapsed = time.time() - start_time
                span.attributes["latency"] = round(elapsed, 4)
                self.tracer.end_span(span)
                return result
            except Exception as exc:
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=getattr(exc, "__traceback__", None),
                )
                self.tracer.end_span(span)
                raise
                
        return wrapper
        
    def _wrap_checkpoint_get(self, original: Callable) -> Callable:
        """Wrap checkpoint load operations."""
        @wraps(original)
        def wrapper(checkpoint_self, *args: Any, **kwargs: Any):  # type: ignore
            span = self.tracer.start_span(
                name="langgraph.checkpoint.get",
                attributes={
                    "service.name": "langgraph",
                    "kind": "checkpoint",
                    "operation": "load",
                },
                tags={"integration": "langgraph"}
            )
            
            start_time = time.time()
            try:
                result = original(checkpoint_self, *args, **kwargs)
                elapsed = time.time() - start_time
                span.attributes["latency"] = round(elapsed, 4)
                self.tracer.end_span(span)
                return result
            except Exception as exc:
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=getattr(exc, "__traceback__", None),
                )
                self.tracer.end_span(span)
                raise
                
        return wrapper

    # ------------------------------------------------------------------
    # Wrapper factory for compiled graph methods
    # ------------------------------------------------------------------
    def _build_graph_wrapper(self, original: Callable, method_name: str = None) -> Callable:  # noqa: C901
        """Generate a patched version of *original* that records a span.

        Supports sync + async callables (invoke / stream / ainvoke / astream).
        """
        is_async = inspect.iscoroutinefunction(original)
        # Use provided method name or try to extract it
        if method_name is None:
            if hasattr(original, '__name__'):
                method_name = original.__name__
            elif hasattr(original, '__func__') and hasattr(original.__func__, '__name__'):
                method_name = original.__func__.__name__
            else:
                method_name = "unknown"

        # ------------------------------------------------------------------
        # Async version
        # ------------------------------------------------------------------
        if is_async:

            async def async_wrapper(compiled_self, *args: Any, **kwargs: Any):  # type: ignore  # noqa: ANN001
                start_time = time.time()
                span = self._start_span(compiled_self, method_name)
                
                # Store current graph span in thread-local for node tracing
                self._thread_local.current_graph_span = span
                
                try:
                    result = await original(compiled_self, *args, **kwargs)

                    if method_name == "astream":
                        return self._wrap_async_generator(result, span, start_time, compiled_self)

                    # Non-stream async call (ainvoke)
                    self._finalise_span(span, start_time, args, kwargs, result)
                    return result
                except Exception as exc:
                    self._record_error(span, exc)
                    raise
                finally:
                    # Clear thread-local
                    self._thread_local.current_graph_span = None

            # Store the original method name on the wrapper
            async_wrapper.__ze_method_name__ = method_name
            return async_wrapper

        # ------------------------------------------------------------------
        # Sync version
        # ------------------------------------------------------------------
        def sync_wrapper(compiled_self, *args: Any, **kwargs: Any):  # type: ignore  # noqa: ANN001
            start_time = time.time()
            span = self._start_span(compiled_self, method_name)
            
            # Store current graph span in thread-local for node tracing
            self._thread_local.current_graph_span = span
            
            try:
                result = original(compiled_self, *args, **kwargs)

                if method_name == "stream":
                    return self._wrap_sync_generator(result, span, start_time, compiled_self)

                # Non-stream call (invoke)
                self._finalise_span(span, start_time, args, kwargs, result)
                return result
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                # Clear thread-local
                self._thread_local.current_graph_span = None

        # Store the original method name on the wrapper
        sync_wrapper.__ze_method_name__ = method_name
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
        for attr in ("name", "id", "graph_id"):
            if hasattr(compiled_self, attr):
                attributes[attr] = getattr(compiled_self, attr)
                break
                
        # Add graph metadata if available
        if hasattr(compiled_self, "_ze_graph_metadata"):
            metadata = compiled_self._ze_graph_metadata
            attributes["node_count"] = len(metadata.get("nodes", []))
            attributes["edge_count"] = len(metadata.get("edges", []))
            attributes["has_conditionals"] = len(metadata.get("conditional_edges", [])) > 0
            
            # Include node names for better visibility
            if metadata.get("nodes"):
                attributes["nodes"] = ",".join(metadata["nodes"][:10])  # Limit to first 10

        return self.tracer.start_span(name=f"langgraph.{method_name}", attributes=attributes, tags={"integration": "langgraph"})

    def _finalise_span(self, span, start_time: float, args, kwargs, output):  # noqa: ANN001
        elapsed = time.time() - start_time
        span.attributes["latency"] = round(elapsed, 4)

        try:
            # Try to extract more meaningful I/O data
            input_data = None
            output_data = None
            
            if args and len(args) > 0:
                # First arg is usually the state/input
                input_data = str(args[0])[:1000]  # Limit size
            elif kwargs:
                # Look for common input keys
                for key in ["messages", "input", "state", "query"]:
                    if key in kwargs:
                        input_data = str(kwargs[key])[:1000]
                        break
                        
            if output:
                # Try to extract output data
                if isinstance(output, dict):
                    # Look for common output keys
                    for key in ["messages", "output", "response", "result"]:
                        if key in output:
                            output_data = str(output[key])[:1000]
                            break
                    if not output_data:
                        output_data = str(output)[:1000]
                else:
                    output_data = str(output)[:1000]
                    
            span.set_io(input_data=input_data or "<no input>", output_data=output_data or "<no output>")
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
    def _wrap_sync_generator(self, gen, span, start_time: float, compiled_self=None):  # noqa: ANN001
        first_chunk_time = None
        node_sequence = []

        def _generator():
            nonlocal first_chunk_time
            try:
                for chunk in gen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["time_to_first_chunk"] = round(first_chunk_time - start_time, 4)
                        
                    # Try to extract node information from streaming chunks
                    try:
                        if isinstance(chunk, dict):
                            # LangGraph often includes node info in stream chunks
                            for key in chunk:
                                if key not in ["__root__", "__start__", "__end__"]:
                                    if key not in node_sequence:
                                        node_sequence.append(key)
                    except Exception:
                        pass
                        
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                # Add node execution sequence to span
                if node_sequence:
                    span.attributes["node_sequence"] = ",".join(node_sequence)
                self._finalise_span(span, start_time, None, None, "<stream>")

        return _generator()

    def _wrap_async_generator(self, agen, span, start_time: float, compiled_self=None):  # noqa: ANN001
        first_chunk_time = None
        node_sequence = []

        async def _async_generator():
            nonlocal first_chunk_time
            try:
                async for chunk in agen:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        span.attributes["time_to_first_chunk"] = round(first_chunk_time - start_time, 4)
                        
                    # Try to extract node information from streaming chunks
                    try:
                        if isinstance(chunk, dict):
                            # LangGraph often includes node info in stream chunks
                            for key in chunk:
                                if key not in ["__root__", "__start__", "__end__"]:
                                    if key not in node_sequence:
                                        node_sequence.append(key)
                    except Exception:
                        pass
                        
                    yield chunk
            except Exception as exc:
                self._record_error(span, exc)
                raise
            finally:
                # Add node execution sequence to span
                if node_sequence:
                    span.attributes["node_sequence"] = ",".join(node_sequence)
                self._finalise_span(span, start_time, None, None, "<stream>")

        return _async_generator() 
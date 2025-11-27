"""
PydanticAI integration for ZeroEval tracing.

This integration patches PydanticAI's Agent class to automatically create spans
for agent executions, ensuring that all LLM calls made during agent.iter() or
agent.run() are grouped under the same trace.

The key insight is that PydanticAI's agent.iter() makes multiple OpenAI API calls
internally (one for each node in the execution graph). Without this integration,
each API call would create a separate trace. This integration creates a parent
span that encompasses the entire agent execution.

For multi-turn conversations, this integration tracks trace IDs across agent runs
by associating them with message history. When the same conversation continues
(identified by shared messages), all runs share the same trace ID.
"""

import logging
import time
import weakref
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional

from ..base import Integration

logger = logging.getLogger(__name__)

# Context variable to track the current agent span across async boundaries
_current_agent_span: ContextVar[Optional[Any]] = ContextVar("pydanticai_agent_span", default=None)

# Global registry mapping message history identity -> trace_id
# Uses WeakValueDictionary concept but we use regular dict with message id() as key
_conversation_trace_registry: dict[int, str] = {}


def _get_conversation_key(message_history: Any) -> Optional[int]:
    """
    Generate a stable key for a conversation based on message history.
    
    We use the id of the first message in the history as the conversation key.
    This works because PydanticAI accumulates messages in the same list object
    across conversation turns.
    """
    if not message_history:
        return None
    try:
        # Use identity of the list itself if it's the same object across calls
        return id(message_history)
    except Exception:
        return None


def _get_trace_id_from_history(message_history: Any) -> Optional[str]:
    """
    Retrieve trace_id associated with a message history if one exists.
    
    Checks both the list identity and the first message identity.
    """
    if not message_history:
        return None
    
    # Try list identity first
    list_key = id(message_history)
    if list_key in _conversation_trace_registry:
        return _conversation_trace_registry[list_key]
    
    # Try first message identity as fallback
    try:
        if len(message_history) > 0:
            first_msg_key = id(message_history[0])
            if first_msg_key in _conversation_trace_registry:
                return _conversation_trace_registry[first_msg_key]
    except Exception:
        pass
    
    return None


def _store_trace_id_for_history(message_history: Any, trace_id: str) -> None:
    """
    Store trace_id associated with message history for future runs.
    
    Stores under both list identity and first message identity for robustness.
    """
    if not message_history or not trace_id:
        return
    
    # Store under list identity
    list_key = id(message_history)
    _conversation_trace_registry[list_key] = trace_id
    
    # Also store under first message identity as fallback
    try:
        if len(message_history) > 0:
            first_msg_key = id(message_history[0])
            _conversation_trace_registry[first_msg_key] = trace_id
    except Exception:
        pass
    
    # Cleanup: limit registry size to prevent memory leak
    if len(_conversation_trace_registry) > 1000:
        # Remove oldest entries (simple FIFO cleanup)
        keys_to_remove = list(_conversation_trace_registry.keys())[:500]
        for k in keys_to_remove:
            _conversation_trace_registry.pop(k, None)


class PydanticAIIntegration(Integration):
    """
    Integration for PydanticAI's Agent class.
    
    Patches Agent methods to automatically create spans for:
    - Agent.iter() - async context manager for streaming agent execution
    - Agent.run() - async method for non-streaming execution
    - Agent.run_sync() - sync wrapper for run()
    
    All LLM calls made during agent execution will be children of the agent span,
    sharing the same trace_id.
    """
    
    PACKAGE_NAME = "pydantic_ai"

    def setup(self) -> None:
        """Set up PydanticAI integration by patching the Agent class."""
        try:
            from pydantic_ai import Agent
            
            logger.debug("[PydanticAI] Setting up integration")
            
            # Patch the iter method (async context manager for streaming)
            if hasattr(Agent, "iter"):
                self._patch_method(Agent, "iter", self._wrap_iter)
                logger.debug("[PydanticAI] Patched Agent.iter")
            
            # Patch the run method (async non-streaming)
            if hasattr(Agent, "run"):
                self._patch_method(Agent, "run", self._wrap_run)
                logger.debug("[PydanticAI] Patched Agent.run")
            
            # Patch run_sync if it exists (sync wrapper)
            if hasattr(Agent, "run_sync"):
                self._patch_method(Agent, "run_sync", self._wrap_run_sync)
                logger.debug("[PydanticAI] Patched Agent.run_sync")
            
            logger.info("[PydanticAI] Integration setup complete")
            
        except Exception as e:
            logger.error(f"[PydanticAI] Failed to setup integration: {e}")
            raise

    def _wrap_iter(self, original: Callable) -> Callable:
        """
        Wrap Agent.iter() to create a parent span for the entire agent execution.
        
        Agent.iter() returns an async context manager that yields nodes.
        We need to wrap it so that:
        1. A span is created when entering the context
        2. The span is ended when exiting the context
        3. All LLM calls during iteration inherit this span's trace_id
        """
        integration = self
        
        @wraps(original)
        def wrapper(agent_self, *args: Any, **kwargs: Any):
            # Get the async context manager from the original method
            original_ctx_manager = original(agent_self, *args, **kwargs)
            
            # Return our wrapped context manager
            return _AgentIterContextManagerWrapper(
                original_ctx_manager=original_ctx_manager,
                agent=agent_self,
                tracer=integration.tracer,
                args=args,
                kwargs=kwargs,
            )
        
        return wrapper

    def _wrap_run(self, original: Callable) -> Callable:
        """
        Wrap Agent.run() to create a parent span for the entire agent execution.
        """
        integration = self
        
        @wraps(original)
        async def wrapper(agent_self, *args: Any, **kwargs: Any):
            start_time = time.time()
            message_history = kwargs.get("message_history")
            
            # Extract agent info for the span
            span_name = _get_agent_span_name(agent_self)
            span_attributes = _build_agent_attributes(agent_self, "run", args, kwargs)
            
            # Check if we have an existing trace_id from previous conversation turns
            existing_trace_id = _get_trace_id_from_history(message_history)
            if existing_trace_id:
                logger.debug(f"[PydanticAI] Reusing trace_id {existing_trace_id} from conversation history")
            
            # Create the parent span for this agent execution
            span = integration.tracer.start_span(
                name=span_name,
                kind="agent",
                attributes=span_attributes,
                tags={"integration": "pydanticai"},
                trace_id=existing_trace_id  # Reuse trace_id if available
            )
            
            # Store the trace_id for this conversation
            if message_history is not None:
                _store_trace_id_for_history(message_history, span.trace_id)
            
            # Store in context var so child spans (OpenAI calls) can find it
            token = _current_agent_span.set(span)
            
            try:
                # Call the original run method
                result = await original(agent_self, *args, **kwargs)
                
                # Record output
                elapsed = time.time() - start_time
                span.attributes["latency"] = round(elapsed, 4)
                
                # Try to extract output data
                try:
                    if hasattr(result, "data"):
                        output_data = result.data
                        if hasattr(output_data, "model_dump_json"):
                            span.attributes["output"] = output_data.model_dump_json()[:1000]
                        else:
                            span.attributes["output"] = str(output_data)[:1000]
                    
                    # Store trace_id for the updated message history
                    if hasattr(result, "all_messages"):
                        updated_messages = result.all_messages()
                        _store_trace_id_for_history(updated_messages, span.trace_id)
                except Exception:
                    pass
                
                integration.tracer.end_span(span)
                return result
                
            except Exception as exc:
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=getattr(exc, "__traceback__", None),
                )
                integration.tracer.end_span(span)
                raise
            finally:
                # Restore context
                _current_agent_span.reset(token)
        
        return wrapper

    def _wrap_run_sync(self, original: Callable) -> Callable:
        """
        Wrap Agent.run_sync() to create a parent span for the entire agent execution.
        """
        integration = self
        
        @wraps(original)
        def wrapper(agent_self, *args: Any, **kwargs: Any):
            start_time = time.time()
            message_history = kwargs.get("message_history")
            
            # Extract agent info for the span
            span_name = _get_agent_span_name(agent_self)
            span_attributes = _build_agent_attributes(agent_self, "run_sync", args, kwargs)
            
            # Check if we have an existing trace_id from previous conversation turns
            existing_trace_id = _get_trace_id_from_history(message_history)
            if existing_trace_id:
                logger.debug(f"[PydanticAI] Reusing trace_id {existing_trace_id} from conversation history")
            
            # Create the parent span for this agent execution
            span = integration.tracer.start_span(
                name=span_name,
                kind="agent",
                attributes=span_attributes,
                tags={"integration": "pydanticai"},
                trace_id=existing_trace_id  # Reuse trace_id if available
            )
            
            # Store the trace_id for this conversation
            if message_history is not None:
                _store_trace_id_for_history(message_history, span.trace_id)
            
            # Store in context var so child spans (OpenAI calls) can find it
            token = _current_agent_span.set(span)
            
            try:
                # Call the original run_sync method
                result = original(agent_self, *args, **kwargs)
                
                # Record output
                elapsed = time.time() - start_time
                span.attributes["latency"] = round(elapsed, 4)
                
                # Try to extract output data
                try:
                    if hasattr(result, "data"):
                        output_data = result.data
                        if hasattr(output_data, "model_dump_json"):
                            span.attributes["output"] = output_data.model_dump_json()[:1000]
                        else:
                            span.attributes["output"] = str(output_data)[:1000]
                    
                    # Store trace_id for the updated message history
                    if hasattr(result, "all_messages"):
                        updated_messages = result.all_messages()
                        _store_trace_id_for_history(updated_messages, span.trace_id)
                except Exception:
                    pass
                
                integration.tracer.end_span(span)
                return result
                
            except Exception as exc:
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=getattr(exc, "__traceback__", None),
                )
                integration.tracer.end_span(span)
                raise
            finally:
                # Restore context
                _current_agent_span.reset(token)
        
        return wrapper


class _AgentIterContextManagerWrapper:
    """
    Wrapper for the async context manager returned by Agent.iter().
    
    This ensures that:
    1. A parent span is created when entering the context
    2. All iterations happen under this parent span
    3. The span is properly closed when the context exits
    4. Multi-turn conversations share the same trace_id
    """
    
    def __init__(
        self,
        original_ctx_manager,
        agent,
        tracer,
        args: tuple,
        kwargs: dict,
    ):
        self._original = original_ctx_manager
        self._agent = agent
        self._tracer = tracer
        self._args = args
        self._kwargs = kwargs
        self._span = None
        self._token = None
        self._start_time = None
        self._agent_run = None  # Will hold the actual agent run object
        self._message_history = kwargs.get("message_history")
    
    async def __aenter__(self):
        """Enter the context and create the parent span."""
        self._start_time = time.time()
        
        # Extract agent info for the span
        span_name = _get_agent_span_name(self._agent)
        span_attributes = _build_agent_attributes(
            self._agent, "iter", self._args, self._kwargs
        )
        
        # Check if we have an existing trace_id from previous conversation turns
        existing_trace_id = _get_trace_id_from_history(self._message_history)
        if existing_trace_id:
            logger.debug(f"[PydanticAI] Reusing trace_id {existing_trace_id} from conversation history")
        
        # Create the parent span for this agent execution
        # Pass existing trace_id to continue the conversation trace
        self._span = self._tracer.start_span(
            name=span_name,
            kind="agent",
            attributes=span_attributes,
            tags={"integration": "pydanticai"},
            trace_id=existing_trace_id  # Reuse trace_id if available
        )
        
        # Store the trace_id for this conversation (for new conversations)
        if self._message_history is not None:
            _store_trace_id_for_history(self._message_history, self._span.trace_id)
        
        # Store in context var so child spans (OpenAI calls) can find it
        self._token = _current_agent_span.set(self._span)
        
        # Enter the original context manager
        self._agent_run = await self._original.__aenter__()
        return self._agent_run
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and close the span."""
        try:
            # Exit the original context manager first
            result = await self._original.__aexit__(exc_type, exc_val, exc_tb)
            
            if self._span:
                elapsed = time.time() - self._start_time
                self._span.attributes["latency"] = round(elapsed, 4)
                
                # Try to capture the final result if available
                try:
                    if self._agent_run and hasattr(self._agent_run, "result"):
                        agent_result = self._agent_run.result
                        if agent_result and hasattr(agent_result, "data"):
                            output_data = agent_result.data
                            if hasattr(output_data, "model_dump_json"):
                                self._span.attributes["output"] = output_data.model_dump_json()[:1000]
                            else:
                                self._span.attributes["output"] = str(output_data)[:1000]
                        
                        # Store trace_id for the updated message history after run
                        if hasattr(agent_result, "all_messages"):
                            updated_messages = agent_result.all_messages()
                            _store_trace_id_for_history(updated_messages, self._span.trace_id)
                except Exception:
                    pass
                
                if exc_type is not None:
                    self._span.set_error(
                        code=exc_type.__name__ if exc_type else "UnknownError",
                        message=str(exc_val) if exc_val else "Unknown error",
                        stack=exc_tb,
                    )
                
                self._tracer.end_span(self._span)
            
            return result
            
        finally:
            # Always restore context
            if self._token is not None:
                _current_agent_span.reset(self._token)


def _get_agent_span_name(agent) -> str:
    """Generate a span name for the agent execution."""
    # Try to get a meaningful name from the agent
    agent_name = None
    
    # Check for name attribute
    if hasattr(agent, "name") and agent.name:
        agent_name = agent.name
    # Check for __name__ on the class
    elif hasattr(agent, "__class__"):
        agent_name = agent.__class__.__name__
    
    if agent_name and agent_name != "Agent":
        return f"pydanticai.agent.{agent_name}"
    
    return "pydanticai.agent"


def _build_agent_attributes(agent, method_name: str, args: tuple, kwargs: dict) -> dict:
    """Build span attributes for an agent execution."""
    attributes = {
        "service.name": "pydanticai",
        "method": method_name,
    }
    
    # Extract model info if available
    if hasattr(agent, "model"):
        model = agent.model
        if isinstance(model, str):
            attributes["model"] = model
        elif hasattr(model, "model_name"):
            attributes["model"] = model.model_name
        elif hasattr(model, "name"):
            attributes["model"] = model.name
        else:
            attributes["model"] = str(model)
    
    # Extract agent name if available
    if hasattr(agent, "name") and agent.name:
        attributes["agent_name"] = agent.name
    
    # Extract output type if available
    if hasattr(agent, "output_type") and agent.output_type:
        try:
            if hasattr(agent.output_type, "__name__"):
                attributes["output_type"] = agent.output_type.__name__
            else:
                attributes["output_type"] = str(agent.output_type)
        except Exception:
            pass
    
    # Extract system prompt info
    if hasattr(agent, "system_prompt") and agent.system_prompt:
        try:
            prompt = agent.system_prompt
            if callable(prompt):
                # It's a function, note that
                attributes["has_dynamic_system_prompt"] = True
            elif isinstance(prompt, str):
                # Store a preview
                attributes["system_prompt_preview"] = prompt[:200]
        except Exception:
            pass
    
    # Extract user input from args/kwargs
    try:
        if args:
            # First positional arg is typically the user prompt
            user_input = args[0]
            if isinstance(user_input, str):
                attributes["input"] = user_input[:500]
        elif "user_prompt" in kwargs:
            user_input = kwargs["user_prompt"]
            if isinstance(user_input, str):
                attributes["input"] = user_input[:500]
    except Exception:
        pass
    
    # Check for tools
    if hasattr(agent, "tools") and agent.tools:
        try:
            tool_names = []
            for tool in agent.tools:
                if hasattr(tool, "name"):
                    tool_names.append(tool.name)
                elif hasattr(tool, "__name__"):
                    tool_names.append(tool.__name__)
            if tool_names:
                attributes["tools"] = ",".join(tool_names[:10])
                attributes["tool_count"] = len(agent.tools)
        except Exception:
            pass
    
    return attributes


def get_current_agent_span():
    """
    Get the current PydanticAI agent span from context.
    
    This can be used by other integrations (like OpenAI) to check if they're
    running within a PydanticAI agent context and inherit the trace ID.
    """
    return _current_agent_span.get()


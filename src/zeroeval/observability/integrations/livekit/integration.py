import asyncio
import json
import logging
import os
import sys
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..base import Integration

if TYPE_CHECKING:
    from livekit.agents import AgentSession

logger = logging.getLogger(__name__)

# Check if debug mode is enabled
debug_mode = os.environ.get("ZEROEVAL_DEBUG", "").lower() in ("true", "1", "yes")
if debug_mode:
    logger.info("LiveKit integration: Debug mode is enabled")


class LiveKitIntegration(Integration):
    """Integration for LiveKit Agents framework.
    
    This integration automatically instruments:
    - Session lifecycle (one session per voice conversation)
    - Agent and human turns (one trace per turn)
    - Within each turn: LLM calls, tool usage, TTS generation, STT processing
    - Performance metrics and signals
    
    The hierarchy is:
    - Session: Entire voice conversation
      - Trace: Each conversational turn
        - Spans: Individual operations (LLM, TTS, STT, tools)
    """
    
    PACKAGE_NAME = "livekit.agents"
    
    def __init__(self, tracer):
        super().__init__(tracer)
        self._original_agent_session = None
        self._original_agent = None
        self._original_job_context = None
        self._original_function_tool = None
        self._original_log_metrics = None
        self._completion_locks = {}  # Lock per speech_id to prevent concurrent completions
        self._ended_span_ids = set()  # Track which span IDs have been ended
    
    def setup(self) -> None:
        """Setup the LiveKit integration by patching key classes and methods."""
        try:
            logger.info("LiveKit integration setup() called")
            
            # Try to patch immediately if modules are already loaded
            self._try_patch_all()
            
            # Also set up a lazy patching mechanism for when modules are imported later
            self._setup_lazy_patching()
            
            logger.info("LiveKit integration setup() completed")
            
        except Exception as e:
            logger.error(f"Failed to setup LiveKit integration: {e}", exc_info=True)
            raise
    
    def _try_patch_all(self):
        """Try to patch all LiveKit components if they're available."""
        # Patch AgentSession
        try:
            from livekit.agents import AgentSession
            if not self._is_already_patched(AgentSession, '__init__'):
                self._patch_agent_session_class(AgentSession)
                logger.info("Patched AgentSession")
        except ImportError:
            logger.debug("AgentSession not yet available")
        
        # Patch Agent
        try:
            from livekit.agents import Agent
            if not self._is_already_patched(Agent, '__init__'):
                self._patch_agent_class(Agent)
                logger.info("Patched Agent")
        except ImportError:
            logger.debug("Agent not yet available")
        
        # Patch JobContext
        try:
            from livekit.agents import JobContext
            if not self._is_already_patched(JobContext, 'connect'):
                self._patch_job_context_class(JobContext)
                logger.info("Patched JobContext")
        except ImportError:
            logger.debug("JobContext not yet available")
        
        # Patch function_tool
        try:
            from livekit.agents.llm import function_tool
            import livekit.agents.llm as llm_module
            if not hasattr(function_tool, '_ze_patched'):
                self._patch_function_tool_module(llm_module)
                logger.info("Patched function_tool")
        except ImportError:
            logger.debug("function_tool not yet available")
        
        # Patch metrics
        try:
            from livekit.agents import metrics
            if hasattr(metrics, 'log_metrics') and not hasattr(metrics.log_metrics, '_ze_patched'):
                self._patch_metrics_module(metrics)
                logger.info("Patched metrics module")
        except ImportError:
            logger.debug("metrics module not yet available")
    
    def _setup_lazy_patching(self):
        """Set up lazy patching that triggers when classes are actually used."""
        # Patch the livekit.agents module's __getattr__ if possible
        try:
            import livekit.agents as agents_module
            original_getattr = getattr(agents_module, '__getattr__', None)
            integration = self
            
            def patched_getattr(name):
                # Try to patch when the class is accessed
                if name == 'AgentSession':
                    integration._try_patch_all()
                
                # Call original getattr if it exists
                if original_getattr:
                    return original_getattr(name)
                else:
                    raise AttributeError(f"module 'livekit.agents' has no attribute '{name}'")
            
            agents_module.__getattr__ = patched_getattr
        except ImportError:
            pass
    
    def _is_already_patched(self, cls, method_name):
        """Check if a method is already patched."""
        method = getattr(cls, method_name, None)
        return method and hasattr(method, '_ze_patched')
    
    def _patch_agent_session_class(self, AgentSession) -> None:
        """Patch AgentSession class methods."""
        # Store originals
        if not self._original_agent_session:
            self._original_agent_session = {
                '__init__': AgentSession.__init__,
                'start': AgentSession.start,
            }
        
        integration = self  # Capture self for closures
        
        @wraps(self._original_agent_session['__init__'])
        def wrapped_init(self_session, *args, **kwargs):
            # Generate session ID that will be used for the entire conversation
            session_id = str(uuid.uuid4())
            self_session._ze_session_id = session_id
            self_session._ze_current_turn_trace_id = None  # Track current turn
            self_session._ze_turn_spans = []  # Spans for current turn
            
            logger.info(f"LiveKit: AgentSession.__init__ called, session_id={session_id}")
            
            # Call original init
            result = integration._original_agent_session['__init__'](self_session, *args, **kwargs)
            
            # Store session metadata for later use when creating spans
            self_session._ze_session_name = f"LiveKit Voice Session - {kwargs.get('room', {}).name if kwargs.get('room') else 'Unknown Room'}"
            self_session._ze_session_tags = {
                "integration": "livekit",
                "llm": kwargs.get("llm", None).__class__.__name__ if kwargs.get("llm") else None,
                "stt": kwargs.get("stt", None).__class__.__name__ if kwargs.get("stt") else None,
                "tts": kwargs.get("tts", None).__class__.__name__ if kwargs.get("tts") else None,
            }
            
            # Patch the instance's on method to intercept event handlers
            integration._patch_session_instance_methods(self_session)
            
            # Also try to patch other components now that we know LiveKit is being used
            integration._try_patch_all()
            
            return result
        
        @wraps(self._original_agent_session['start'])
        async def wrapped_start(self_session, *args, **kwargs):
            session_id = getattr(self_session, "_ze_session_id", str(uuid.uuid4()))
            
            logger.info(f"LiveKit: AgentSession.start called, session_id={session_id}")
            
            # Create a span for session start
            span = integration.tracer.start_span(
                name="livekit.session.start",
                attributes={
                    "service.name": "livekit",
                    "kind": "session_start",
                    "room.name": kwargs.get("room", None).name if kwargs.get("room") else None,
                    "agent.class": kwargs.get("agent", None).__class__.__name__ if kwargs.get("agent") else None,
                },
                tags={"integration": "livekit"},
                session_id=session_id,
                session_name=self_session._ze_session_name,
            )
            
            # Apply session tags
            integration.tracer.add_session_tags(session_id, self_session._ze_session_tags)
            
            try:
                result = await integration._original_agent_session['start'](self_session, *args, **kwargs)
                logger.info(f"LiveKit: Session started successfully for {session_id}")
                if span.span_id not in integration._ended_span_ids:
                    integration._ended_span_ids.add(span.span_id)
                    integration.tracer.end_span(span)
                return result
            except Exception as e:
                logger.error(f"LiveKit: Session start failed for {session_id}: {e}")
                span.set_error(
                    code=type(e).__name__,
                    message=str(e),
                    stack=getattr(e, "__traceback__", None),
                )
                if span.span_id not in integration._ended_span_ids:
                    integration._ended_span_ids.add(span.span_id)
                    integration.tracer.end_span(span)
                raise
        
        # Apply patches
        wrapped_init._ze_patched = True
        wrapped_start._ze_patched = True
        AgentSession.__init__ = wrapped_init
        AgentSession.start = wrapped_start
    
    def _patch_session_instance_methods(self, session_instance) -> None:
        """Patch instance methods of an AgentSession to intercept events."""
        if not hasattr(session_instance, 'on'):
            return
            
        # Patch the on method of this specific instance
        original_on = session_instance.on
        integration = self  # Capture self for closure
        
        @wraps(original_on)
        def wrapped_on(event_name, handler=None):
            logger.debug(f"LiveKit: Registering event handler for {event_name}")
            
            if handler is None:
                # It's being used as a decorator
                def decorator(func):
                    wrapped_func = integration._wrap_event_handler(event_name, func, session_instance)
                    return original_on(event_name)(wrapped_func)
                return decorator
            else:
                # Direct call with handler
                wrapped_handler = integration._wrap_event_handler(event_name, handler, session_instance)
                return original_on(event_name, wrapped_handler)
        
        session_instance.on = wrapped_on
        
        # Register our own event handlers to capture additional telemetry
        self._register_default_event_handlers(session_instance)
    
    def _register_default_event_handlers(self, session_instance) -> None:
        """Register default event handlers to capture LiveKit telemetry."""
        integration = self
        
        # Listen for speech events
        @session_instance.on("speech_created")
        def on_speech_created(event):
            logger.debug(f"LiveKit: speech_created event")
            # Speech created events indicate the start of agent speech
        
        @session_instance.on("user_speech_created") 
        def on_user_speech_created(event):
            logger.debug(f"LiveKit: user_speech_created event")
            # User speech created events indicate the start of user speech
            
        @session_instance.on("user_speech_committed")
        def on_user_speech_committed(event):
            logger.debug(f"LiveKit: user_speech_committed event: {event}")
            # User speech committed indicates the end of user turn
            
        @session_instance.on("function_called")
        def on_function_called(event):
            logger.debug(f"LiveKit: function_called event")
            # Function called events for tool usage
    
    def _wrap_event_handler(self, event_name: str, handler: Callable, session_instance) -> Callable:
        """Wrap an event handler to capture metrics and events."""
        @wraps(handler)
        def wrapped_handler(event):
            logger.debug(f"LiveKit: Event {event_name} triggered")
            
            # Handle metrics collection events to track turns
            if event_name == "metrics_collected":
                self._capture_metrics_in_turn(event, session_instance)
            
            # Call the original handler
            try:
                result = handler(event)
                return result
            except Exception as e:
                logger.error(f"Error in event handler {event_name}: {e}")
                raise
        
        return wrapped_handler
    
    def _get_or_create_turn(self, session_instance, speech_id: str, turn_type: str) -> None:
        """Get or create a turn trace for the given speech_id."""
        # Check if we already have a trace for this speech_id
        if not hasattr(session_instance, "_ze_speech_traces"):
            session_instance._ze_speech_traces = {}
        
        if speech_id in session_instance._ze_speech_traces:
            # Already tracking this turn
            return speech_id
        
        # Create a new trace for this turn
        session_id = getattr(session_instance, "_ze_session_id", "unknown")
        
        logger.info(f"LiveKit: Starting new {turn_type} turn for speech_id={speech_id}")
        
        # Create the root span for this turn
        span = self.tracer.start_span(
            name=f"livekit.turn.{turn_type}",
            attributes={
                "service.name": "livekit",
                "kind": "turn",
                "turn.type": turn_type,
                "speech.id": speech_id,
            },
            tags={"integration": "livekit", "turn_type": turn_type},
            session_id=session_id,
            session_name=getattr(session_instance, "_ze_session_name", None),
        )
        
        # Store the turn info
        session_instance._ze_speech_traces[speech_id] = {
            "root_span": span,  # Store the root span for this turn
            "turn_type": turn_type,
            "spans": [span],
            "start_time": time.time(),
        }
        
        return speech_id  # Return speech_id instead of trace_id
    
    def _capture_metrics_in_turn(self, event, session_instance) -> None:
        """Capture metrics within the appropriate turn."""
        try:
            metrics = event.metrics if hasattr(event, "metrics") else event
            
            # Determine turn type and speech_id
            metrics_type = metrics.__class__.__name__
            speech_id = getattr(metrics, "speech_id", None)
            
            # Determine turn type based on metrics
            if metrics_type in ["STTMetrics", "EOUMetrics"]:
                turn_type = "user"
            elif metrics_type in ["LLMMetrics", "TTSMetrics"]:
                turn_type = "agent"
            else:
                # VAD metrics don't belong to a specific turn
                return
            
            # Use speech_id or create a temporary one
            if not speech_id:
                # For metrics without speech_id, use a time-based ID
                speech_id = f"turn_{int(time.time() * 1000)}"
            
            # Get or create the turn trace
            speech_id_result = self._get_or_create_turn(session_instance, speech_id, turn_type)
            
            # Process the metrics within this turn
            self._process_metrics_in_turn(metrics, session_instance, speech_id)
            
            # Check if turn might be complete
            # For user turns: complete after EOUMetrics
            # For agent turns: complete after TTSMetrics
            if (turn_type == "user" and metrics_type == "EOUMetrics") or \
               (turn_type == "agent" and metrics_type == "TTSMetrics"):
                # Schedule turn completion after a delay
                self._schedule_turn_completion(session_instance, speech_id)
                
        except Exception as e:
            logger.error(f"Failed to capture metrics: {e}")
    
    def _schedule_turn_completion(self, session_instance, speech_id: str) -> None:
        """Schedule completion of a turn after TTS is done."""
        # Complete any previous turns first
        if hasattr(session_instance, "_ze_speech_traces"):
            for sid in list(session_instance._ze_speech_traces.keys()):
                if sid != speech_id:
                    # Use asyncio to schedule completion
                    asyncio.create_task(self._complete_turn_async(session_instance, sid))
        
        # Complete current turn after a short delay (100ms)
        # This ensures we capture all metrics for the turn
        if hasattr(session_instance, "_ze_speech_traces") and speech_id in session_instance._ze_speech_traces:
            turn_info = session_instance._ze_speech_traces[speech_id]
            
            # Cancel any existing task for this turn
            if "completion_task" in turn_info and turn_info["completion_task"] and not turn_info["completion_task"].done():
                turn_info["completion_task"].cancel()
            
            async def complete_after_delay():
                await asyncio.sleep(0.1)  # 100ms delay
                await self._complete_turn_async(session_instance, speech_id)
            
            # Schedule completion using asyncio
            turn_info["completion_task"] = asyncio.create_task(complete_after_delay())
    
    async def _complete_turn_async(self, session_instance, speech_id: str) -> None:
        """Complete a turn by ending all its spans (async version with locking)."""
        # Get or create a lock for this speech_id
        if speech_id not in self._completion_locks:
            self._completion_locks[speech_id] = asyncio.Lock()
        
        async with self._completion_locks[speech_id]:
            # Double-check the turn still exists
            if not hasattr(session_instance, "_ze_speech_traces") or speech_id not in session_instance._ze_speech_traces:
                return
            
            turn_info = session_instance._ze_speech_traces[speech_id]
            logger.info(f"LiveKit: Completing turn for speech_id={speech_id}")
            
            # Cancel any pending completion task
            if "completion_task" in turn_info and turn_info["completion_task"] and not turn_info["completion_task"].done():
                turn_info["completion_task"].cancel()
            
            # End all spans in reverse order, but only if not already ended
            for span in reversed(turn_info["spans"]):
                if span.span_id not in self._ended_span_ids:
                    self._ended_span_ids.add(span.span_id)
                    self.tracer.end_span(span)
                else:
                    logger.debug(f"LiveKit: Skipping already ended span {span.span_id}")
            
            # Remove from active traces
            del session_instance._ze_speech_traces[speech_id]
            
            # Clean up the lock
            del self._completion_locks[speech_id]
    
    def _complete_turn(self, session_instance, speech_id: str) -> None:
        """Complete a turn by ending all its spans (sync wrapper for backwards compatibility)."""
        asyncio.create_task(self._complete_turn_async(session_instance, speech_id))
    
    def _patch_agent_class(self, Agent) -> None:
        """Patch Agent class methods."""
        # Store original
        if not self._original_agent:
            self._original_agent = Agent.__init__
        
        integration = self
        
        @wraps(self._original_agent)
        def wrapped_init(self_agent, *args, **kwargs):
            agent_id = str(uuid.uuid4())
            self_agent._ze_agent_id = agent_id
            
            logger.info(f"LiveKit: Agent.__init__ called, agent_id={agent_id}")
            
            # We don't create a span here as this is part of session setup
            # Just store the agent info for later use
            
            try:
                result = integration._original_agent(self_agent, *args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Agent init failed: {e}")
                raise
        
        wrapped_init._ze_patched = True
        Agent.__init__ = wrapped_init
    
    def _patch_job_context_class(self, JobContext) -> None:
        """Patch JobContext class methods."""
        # Store original
        if not self._original_job_context:
            self._original_job_context = JobContext.connect
        
        integration = self
        
        @wraps(self._original_job_context)
        async def wrapped_connect(self_ctx, *args, **kwargs):
            # JobContext connect is part of session setup, not a separate operation
            logger.info(f"LiveKit: JobContext.connect called")
            
            try:
                result = await integration._original_job_context(self_ctx, *args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"JobContext connect failed: {e}")
                raise
        
        wrapped_connect._ze_patched = True
        JobContext.connect = wrapped_connect
    
    def _patch_function_tool_module(self, llm_module) -> None:
        """Patch function_tool decorator on the module."""
        if not self._original_function_tool:
            self._original_function_tool = llm_module.function_tool
        
        integration = self  # Capture self for closure
        
        def wrapped_decorator(func=None, *, name=None, description=None):
            def actual_decorator(f):
                # Apply original decorator
                if func is None:
                    decorated = integration._original_function_tool(f, name=name, description=description)
                else:
                    decorated = integration._original_function_tool(f)
                
                @wraps(decorated)
                async def wrapper(*args, **kwargs):
                    tool_name = name or f.__name__
                    logger.info(f"LiveKit: function_tool {tool_name} called")
                    
                    # Find the session from the RunContext (first argument)
                    session_instance = None
                    if args and hasattr(args[0], 'session'):
                        session_instance = args[0].session
                    
                    # Find the most recent agent turn to attach this tool call to
                    if session_instance and hasattr(session_instance, "_ze_speech_traces"):
                        # Find the most recent agent turn
                        recent_agent_turn = None
                        recent_time = 0
                        
                        for speech_id, turn_info in session_instance._ze_speech_traces.items():
                            if turn_info["turn_type"] == "agent" and turn_info["start_time"] > recent_time:
                                recent_agent_turn = (speech_id, turn_info)
                                recent_time = turn_info["start_time"]
                        
                        if recent_agent_turn:
                            speech_id, turn_info = recent_agent_turn
                            root_span = turn_info["root_span"]
                            
                            # Create tool span as a child of the turn span
                            span = integration.tracer.start_span(
                                name=f"livekit.tool.{tool_name}",
                                attributes={
                                    "service.name": "livekit",
                                    "kind": "tool",
                                    "tool.name": tool_name,
                                    "tool.description": description or f.__doc__ or "",
                                },
                                tags={"integration": "livekit", "tool_type": "function"},
                                session_id=session_instance._ze_session_id,
                                session_name=getattr(session_instance, "_ze_session_name", None),
                            )
                            
                            # Manually set parent relationship
                            span.parent_id = root_span.span_id
                            span.trace_id = root_span.trace_id
                            
                            turn_info["spans"].append(span)
                            
                            try:
                                result = await decorated(*args, **kwargs)
                                span.set_io(
                                    input_data=json.dumps({"args": str(args[1:]), "kwargs": str(kwargs)}),
                                    output_data=str(result) if result else None,
                                )
                                return result
                            except Exception as e:
                                span.set_error(
                                    code=type(e).__name__,
                                    message=str(e),
                                    stack=getattr(e, "__traceback__", None),
                                )
                                raise
                            finally:
                                if span.span_id not in integration._ended_span_ids:
                                    integration._ended_span_ids.add(span.span_id)
                                    integration.tracer.end_span(span)
                        else:
                            # No active turn, just call the function
                            return await decorated(*args, **kwargs)
                    else:
                        # Outside of session context, just call the function
                        return await decorated(*args, **kwargs)
                
                return wrapper
            
            if func is None:
                return actual_decorator
            else:
                return actual_decorator(func)
        
        wrapped_decorator._ze_patched = True
        llm_module.function_tool = wrapped_decorator
    
    def _patch_metrics_module(self, metrics_module) -> None:
        """Patch the metrics module."""
        if not self._original_log_metrics:
            self._original_log_metrics = metrics_module.log_metrics
        
        integration = self
        
        @wraps(self._original_log_metrics)
        def wrapped_log_metrics(metrics_obj):
            logger.debug(f"LiveKit: log_metrics called with {metrics_obj.__class__.__name__}")
            # We don't capture metrics here as they're already captured via events
            # Just call original
            return integration._original_log_metrics(metrics_obj)
        
        wrapped_log_metrics._ze_patched = True
        metrics_module.log_metrics = wrapped_log_metrics
    
    def _process_metrics_in_turn(self, metrics, session_instance, speech_id: str) -> None:
        """Process metrics and create appropriate spans within the current turn."""
        metrics_type = metrics.__class__.__name__
        session_id = session_instance._ze_session_id
        turn_info = session_instance._ze_speech_traces[speech_id]
        root_span = turn_info["root_span"]
        
        logger.info(f"LiveKit: Processing metrics of type {metrics_type} in turn {speech_id}")
        
        # Map metrics types to appropriate span names
        span_name_map = {
            "STTMetrics": "livekit.stt",
            "LLMMetrics": "livekit.llm", 
            "TTSMetrics": "livekit.tts",
            "EOUMetrics": "livekit.eou",
            "VADMetrics": None,  # VAD metrics are too noisy, skip them
        }
        
        span_name = span_name_map.get(metrics_type)
        if not span_name:
            return  # Skip this metric type
        
        # Create a child span under the root turn span
        # To do this, we need to ensure the root span is on the active span stack
        # Since we can't use a context manager here, we'll set the parent_id directly
        span = self.tracer.start_span(
            name=span_name,
            attributes={
                "service.name": "livekit",
                "kind": metrics_type.lower().replace("metrics", ""),
            },
            tags={"integration": "livekit", "metrics_type": metrics_type},
            session_id=session_id,
            session_name=getattr(session_instance, "_ze_session_name", None),
        )
        
        # Manually set parent relationship
        span.parent_id = root_span.span_id
        span.trace_id = root_span.trace_id
        
        turn_info["spans"].append(span)
        
        # Extract metrics as attributes
        attributes = {}
        
        # STT Metrics
        if hasattr(metrics, "audio_duration"):
            attributes["stt.audio_duration"] = metrics.audio_duration
        if hasattr(metrics, "duration"):
            attributes["duration"] = metrics.duration
        if hasattr(metrics, "streamed"):
            attributes["stt.streamed"] = metrics.streamed
        
        # LLM Metrics
        if hasattr(metrics, "completion_tokens"):
            attributes["llm.completion_tokens"] = metrics.completion_tokens
        if hasattr(metrics, "prompt_tokens"):
            attributes["llm.prompt_tokens"] = metrics.prompt_tokens
        if hasattr(metrics, "total_tokens"):
            attributes["llm.total_tokens"] = metrics.total_tokens
        if hasattr(metrics, "ttft"):
            attributes["llm.ttft"] = metrics.ttft
            span.set_signal("ttft_ms", metrics.ttft * 1000)
        if hasattr(metrics, "tokens_per_second"):
            attributes["llm.tokens_per_second"] = metrics.tokens_per_second
        
        # TTS Metrics
        if hasattr(metrics, "characters_count"):
            attributes["tts.characters_count"] = metrics.characters_count
        if hasattr(metrics, "ttfb"):
            attributes["tts.ttfb"] = metrics.ttfb
            span.set_signal("ttfb_ms", metrics.ttfb * 1000)
        
        # EOU Metrics
        if hasattr(metrics, "end_of_utterance_delay"):
            attributes["eou.end_of_utterance_delay"] = metrics.end_of_utterance_delay
        if hasattr(metrics, "transcription_delay"):
            attributes["eou.transcription_delay"] = metrics.transcription_delay
        
        # Common attributes
        if hasattr(metrics, "speech_id"):
            attributes["speech_id"] = metrics.speech_id
        
        # Update span with attributes
        span.attributes.update(attributes)
        
        # Record duration signal if available
        if hasattr(metrics, "duration"):
            span.set_signal("duration_ms", metrics.duration * 1000)
        
        # Calculate and record conversation latency if we have all components
        if hasattr(metrics, "end_of_utterance_delay") and hasattr(metrics, "ttft") and hasattr(metrics, "ttfb"):
            total_latency = metrics.end_of_utterance_delay + metrics.ttft + metrics.ttfb
            span.set_signal("conversation_latency_ms", total_latency * 1000)
        
        # End the span immediately after setting all attributes
        if span.span_id not in self._ended_span_ids:
            self._ended_span_ids.add(span.span_id)
            self.tracer.end_span(span) 
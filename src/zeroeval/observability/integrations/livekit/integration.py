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
        self._original_agent_activity = None
        self._original_function_tool = None
        self._ended_span_ids = set()  # Track which span IDs have been ended
    
    @property
    def name(self) -> str:
        return "LiveKitIntegration"
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if LiveKit Agents is available."""
        try:
            import livekit.agents
            return True
        except ImportError:
            return False
    
    def setup(self) -> None:
        """Setup the LiveKit integration."""
        self._try_patch_all()
    
    def _try_patch_all(self):
        """Try to patch all LiveKit components."""
        logger.info(f"LiveKit integration: Setting up patches...")
        
        try:
            import livekit.agents
            from livekit.agents import AgentSession
            from livekit.agents.voice.agent_activity import AgentActivity
            from livekit.agents.llm import function_tool
            
            # Patch classes
            self._patch_agent_session_class(AgentSession)
            self._patch_agent_activity_class(AgentActivity)
            self._patch_function_tool_module(livekit.agents.llm)
            
            logger.info("LiveKit integration: All patches applied successfully")
            
        except ImportError as e:
            logger.warning(f"LiveKit integration: Failed to import modules: {e}")
        except Exception as e:
            logger.error(f"LiveKit integration: Error during patching: {e}")
    
    def _patch_agent_session_class(self, AgentSession) -> None:
        """Patch AgentSession class methods."""
        # Store originals
        if not self._original_agent_session:
            self._original_agent_session = {
                '__init__': AgentSession.__init__,
                'start': AgentSession.start,
                '_update_user_state': AgentSession._update_user_state,
            }
        
        integration = self  # Capture self for closures
        
        @wraps(self._original_agent_session['__init__'])
        def wrapped_init(self_session, *args, **kwargs):
            # Generate session ID that will be used for the entire conversation
            session_id = str(uuid.uuid4())
            self_session._ze_session_id = session_id
            self_session._ze_speech_traces = {}  # Track traces by speech_id
            
            logger.info(f"LiveKit: AgentSession.__init__ called, session_id={session_id}")
            
            # Call original init
            result = integration._original_agent_session['__init__'](self_session, *args, **kwargs)
            
            # Store session metadata for later use when creating spans
            room_name = "Unknown Room"
            if 'room' in kwargs and kwargs['room'] and hasattr(kwargs['room'], 'name'):
                room_name = kwargs['room'].name
            
            self_session._ze_session_name = f"LiveKit Voice Session - {room_name}"
            self_session._ze_session_tags = {"integration": "livekit"}
            
            # Safely add component tags only if they have valid class names
            llm = kwargs.get("llm")
            if llm and hasattr(llm, "__class__") and hasattr(llm.__class__, "__name__"):
                llm_name = llm.__class__.__name__
                if llm_name:  # Only add if the name is not None or empty
                    self_session._ze_session_tags["llm"] = llm_name
            
            stt = kwargs.get("stt")
            if stt and hasattr(stt, "__class__") and hasattr(stt.__class__, "__name__"):
                stt_name = stt.__class__.__name__
                if stt_name:
                    self_session._ze_session_tags["stt"] = stt_name
                    
            tts = kwargs.get("tts")
            if tts and hasattr(tts, "__class__") and hasattr(tts.__class__, "__name__"):
                tts_name = tts.__class__.__name__
                if tts_name:
                    self_session._ze_session_tags["tts"] = tts_name
            
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
                
                # Now patch instance methods after session is started
                if hasattr(self_session, '_activity') and self_session._activity:
                    integration._patch_activity_instance(self_session._activity, self_session)
                
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
        
        @wraps(self._original_agent_session['_update_user_state'])
        def wrapped_update_user_state(self_session, state, *, last_speaking_time=None):
            """Wrapped _update_user_state to track user turns."""
            session_id = getattr(self_session, "_ze_session_id", None)
            old_state = getattr(self_session, "_user_state", "listening")
            
            # Track user turn start
            if state == "speaking" and old_state != "speaking" and session_id:
                # Generate a unique speech ID for this user turn
                speech_id = f"user_turn_{int(time.time() * 1000)}"
                
                # Check if we already have a user turn active (shouldn't happen but be safe)
                if not hasattr(self_session, "_ze_user_turn_span") or not self_session._ze_user_turn_span:
                    logger.info(f"LiveKit: Starting user turn for speech_id={speech_id}")
                    
                    turn_span = integration.tracer.start_span(
                        name="livekit.turn.user",
                        attributes={
                            "service.name": "livekit",
                            "kind": "turn",
                            "turn.type": "user",
                            "speech.id": speech_id,
                        },
                        tags={"integration": "livekit", "turn_type": "user"},
                        session_id=session_id,
                        session_name=getattr(self_session, "_ze_session_name", None),
                        is_new_trace=True
                    )
                    
                    # Store the user turn span
                    self_session._ze_user_turn_span = turn_span
                    self_session._ze_user_turn_speech_id = speech_id
                    self_session._ze_current_turn_span = turn_span
            
            # Call original method
            result = integration._original_agent_session['_update_user_state'](
                self_session, state, last_speaking_time=last_speaking_time
            )
            
            # Track user turn end
            if old_state == "speaking" and state in ["listening", "away"] and session_id:
                if hasattr(self_session, "_ze_user_turn_span") and self_session._ze_user_turn_span:
                    turn_span = self_session._ze_user_turn_span
                    speech_id = getattr(self_session, "_ze_user_turn_speech_id", "unknown")
                    
                    logger.info(f"LiveKit: Ending user turn for speech_id={speech_id}")
                    
                    if turn_span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(turn_span.span_id)
                        integration.tracer.end_span(turn_span)
                    
                    # Clean up
                    self_session._ze_user_turn_span = None
                    self_session._ze_user_turn_speech_id = None
                    if hasattr(self_session, '_ze_current_turn_span'):
                        delattr(self_session, '_ze_current_turn_span')
            
            return result
        
        # Apply patches
        wrapped_init._ze_patched = True
        wrapped_start._ze_patched = True
        wrapped_update_user_state._ze_patched = True
        AgentSession.__init__ = wrapped_init
        AgentSession.start = wrapped_start
        AgentSession._update_user_state = wrapped_update_user_state
    
    def _patch_agent_activity_class(self, AgentActivity) -> None:
        """Patch AgentActivity class methods to intercept turn creation."""
        if not self._original_agent_activity:
            self._original_agent_activity = {
                '_pipeline_reply_task': AgentActivity._pipeline_reply_task,
                '_realtime_generation_task': getattr(AgentActivity, '_realtime_generation_task', None),
            }
        
        integration = self
        
        # Patch the assistant turn method
        original_pipeline_reply = self._original_agent_activity['_pipeline_reply_task']
        
        @wraps(original_pipeline_reply)
        async def wrapped_pipeline_reply_task(self_activity, *args, **kwargs):
            # Extract speech_handle from kwargs
            speech_handle = kwargs.get('speech_handle')
            speech_id = speech_handle.id if speech_handle else f"turn_{int(time.time() * 1000)}"
            
            # Get session from activity
            session = self_activity._session
            session_id = getattr(session, "_ze_session_id", None)
            
            if session_id:
                logger.info(f"LiveKit: Starting assistant turn for speech_id={speech_id}")
                
                # Create a new trace for this turn
                turn_span = integration.tracer.start_span(
                    name="livekit.turn.agent",
                    attributes={
                        "service.name": "livekit",
                        "kind": "turn",
                        "turn.type": "agent",
                        "speech.id": speech_id,
                    },
                    tags={"integration": "livekit", "turn_type": "agent"},
                    session_id=session_id,
                    session_name=getattr(session, "_ze_session_name", None),
                    is_new_trace=True  # This creates a new trace under the session
                )
                
                # Store turn span for nested operations to reference
                session._ze_current_turn_span = turn_span
                session._ze_speech_traces[speech_id] = turn_span
            
            try:
                # Call original method
                result = await original_pipeline_reply(self_activity, *args, **kwargs)
                return result
            finally:
                # End the turn span
                if session_id and speech_id in session._ze_speech_traces:
                    turn_span = session._ze_speech_traces[speech_id]
                    if turn_span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(turn_span.span_id)
                        integration.tracer.end_span(turn_span)
                    del session._ze_speech_traces[speech_id]
                    if hasattr(session, '_ze_current_turn_span'):
                        delattr(session, '_ze_current_turn_span')
        
        wrapped_pipeline_reply_task._ze_patched = True
        AgentActivity._pipeline_reply_task = wrapped_pipeline_reply_task
        
        # Patch realtime generation task if it exists
        if self._original_agent_activity['_realtime_generation_task']:
            original_realtime = self._original_agent_activity['_realtime_generation_task']
            
            @wraps(original_realtime)
            async def wrapped_realtime_generation_task(self_activity, *args, **kwargs):
                # Similar logic for realtime turns
                speech_handle = kwargs.get('speech_handle')
                speech_id = speech_handle.id if speech_handle else f"turn_{int(time.time() * 1000)}"
                
                session = self_activity._session
                session_id = getattr(session, "_ze_session_id", None)
                
                if session_id:
                    logger.info(f"LiveKit: Starting realtime assistant turn for speech_id={speech_id}")
                    
                    turn_span = integration.tracer.start_span(
                        name="livekit.turn.agent_realtime",
                        attributes={
                            "service.name": "livekit",
                            "kind": "turn",
                            "turn.type": "agent",
                            "realtime": True,
                            "speech.id": speech_id,
                        },
                        tags={"integration": "livekit", "turn_type": "agent"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                        is_new_trace=True
                    )
                    
                    session._ze_current_turn_span = turn_span
                    session._ze_speech_traces[speech_id] = turn_span
                
                try:
                    result = await original_realtime(self_activity, *args, **kwargs)
                    return result
                finally:
                    if session_id and speech_id in session._ze_speech_traces:
                        turn_span = session._ze_speech_traces[speech_id]
                        if turn_span.span_id not in integration._ended_span_ids:
                            integration._ended_span_ids.add(turn_span.span_id)
                            integration.tracer.end_span(turn_span)
                        del session._ze_speech_traces[speech_id]
                        if hasattr(session, '_ze_current_turn_span'):
                            delattr(session, '_ze_current_turn_span')
            
            wrapped_realtime_generation_task._ze_patched = True
            AgentActivity._realtime_generation_task = wrapped_realtime_generation_task
    

    def _patch_activity_instance(self, activity_instance, session_instance) -> None:
        """Patch instance methods of an AgentActivity to intercept events."""
        if not hasattr(activity_instance, 'on'):
            return
            
        # Patch the on method of this specific instance
        original_on = activity_instance.on
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
        
        activity_instance.on = wrapped_on
    
    def _wrap_event_handler(self, event_name: str, handler: Callable, session_instance) -> Callable:
        """Wrap an event handler to capture metrics and events."""
        @wraps(handler)
        def wrapped_handler(event):
            logger.debug(f"LiveKit: Event {event_name} triggered")
            
            # Handle metrics collection events
            if event_name == "metrics_collected":
                self._capture_metrics_span(event, session_instance)
            
            # Call the original handler
            try:
                result = handler(event)
                return result
            except Exception as e:
                logger.error(f"Error in event handler {event_name}: {e}")
                raise
        
        return wrapped_handler
    

    
    def _capture_metrics_span(self, event, session_instance) -> None:
        """Capture metrics as spans within the current turn."""
        try:
            metrics = event.metrics if hasattr(event, "metrics") else event
            metrics_type = metrics.__class__.__name__
            
            # Skip VAD metrics as they're too noisy
            if metrics_type == "VADMetrics":
                return
            
            # Map metrics types to appropriate span names
            span_name_map = {
                "STTMetrics": "livekit.stt",
                "LLMMetrics": "livekit.llm", 
                "TTSMetrics": "livekit.tts",
                "EOUMetrics": "livekit.eou",
            }
            
            span_name = span_name_map.get(metrics_type)
            if not span_name:
                return
            
            # Get current turn span if available
            current_turn_span = getattr(session_instance, "_ze_current_turn_span", None)
            session_id = getattr(session_instance, "_ze_session_id", None)
            
            if not session_id:
                return
            
            # Create a span for this metric
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
            
            # If we have a current turn span, make this a child of it
            if current_turn_span:
                span.parent_id = current_turn_span.span_id
                span.trace_id = current_turn_span.trace_id
            
            # Add metrics attributes
            self._add_metrics_attributes(span, metrics, metrics_type)
            
            # End the span immediately
            if span.span_id not in self._ended_span_ids:
                self._ended_span_ids.add(span.span_id)
                self.tracer.end_span(span)
                
        except Exception as e:
            logger.error(f"Failed to capture metrics: {e}")
    
    def _patch_function_tool_module(self, llm_module) -> None:
        """Patch the function_tool decorator."""
        if hasattr(llm_module, 'function_tool') and not hasattr(llm_module.function_tool, '_ze_patched'):
            self._original_function_tool = llm_module.function_tool
            integration = self
            
            def wrapped_decorator(func=None, *, name=None, description=None):
                def actual_decorator(f):
                    # Apply original decorator
                    if func is not None:
                        decorated = integration._original_function_tool(func, name=name, description=description)
                    else:
                        decorated = integration._original_function_tool(name=name, description=description)(f)
                    
                    @wraps(decorated)
                    async def wrapper(*args, **kwargs):
                        # Try to get session from context
                        session = None
                        for arg in args:
                            if hasattr(arg, 'session'):
                                session = arg.session
                                break
                        
                        if session:
                            session_id = getattr(session, "_ze_session_id", None)
                            current_turn_span = getattr(session, "_ze_current_turn_span", None)
                            
                            if session_id:
                                tool_name = name or f.__name__
                                
                                span = integration.tracer.start_span(
                                    name=f"livekit.tool.{tool_name}",
                                    attributes={
                                        "service.name": "livekit",
                                        "kind": "tool",
                                        "tool.name": tool_name,
                                    },
                                    tags={"integration": "livekit"},
                                    session_id=session_id,
                                    session_name=getattr(session, "_ze_session_name", None),
                                )
                                
                                # Make it a child of current turn if available
                                if current_turn_span:
                                    span.parent_id = current_turn_span.span_id
                                    span.trace_id = current_turn_span.trace_id
                                
                                try:
                                    result = await decorated(*args, **kwargs)
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
                            # No session context, just call the original
                            return await decorated(*args, **kwargs)
                    
                    wrapper._ze_patched = True
                    return wrapper
                
                if func is not None:
                    return actual_decorator(func)
                return actual_decorator
            
            wrapped_decorator._ze_patched = True
            llm_module.function_tool = wrapped_decorator
    
    def _add_metrics_attributes(self, span, metrics, metrics_type: str) -> None:
        """Add metrics data as attributes to the span."""
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
    
    def teardown(self) -> None:
        """Teardown the integration by restoring original methods."""
        # Restore AgentSession
        if self._original_agent_session:
            try:
                from livekit.agents import AgentSession
                AgentSession.__init__ = self._original_agent_session['__init__']
                AgentSession.start = self._original_agent_session['start']
            except Exception:
                pass
        
        # Restore AgentActivity
        if self._original_agent_activity:
            try:
                from livekit.agents.voice.agent_activity import AgentActivity
                if self._original_agent_activity['_pipeline_reply_task']:
                    AgentActivity._pipeline_reply_task = self._original_agent_activity['_pipeline_reply_task']
                if self._original_agent_activity['_realtime_generation_task']:
                    AgentActivity._realtime_generation_task = self._original_agent_activity['_realtime_generation_task']
            except Exception:
                pass
        
        # Restore function_tool
        if self._original_function_tool:
            try:
                import livekit.agents.llm as llm_module
                llm_module.function_tool = self._original_function_tool
            except Exception:
                pass
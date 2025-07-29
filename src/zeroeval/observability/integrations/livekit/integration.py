import asyncio
import json
import logging
import os
import sys
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
import weakref

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
        self._original_deepgram_stt = None
        self._original_deepgram_stream = None
        self._original_cartesia_tts = None
        self._original_cartesia_stream = None
        self._ended_span_ids = set()  # Track which span IDs have been ended
        self._stream_sessions = weakref.WeakValueDictionary()  # Track stream sessions
        self._original_agent_stt_node = None
    
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
            
            # Try to patch Agent's stt_node for Deepgram tracking
            self._patch_agent_stt_node()
            
            # Try to patch Deepgram if available
            self._try_patch_deepgram()
            
            # Try to patch Cartesia if available
            self._try_patch_cartesia()
            
            logger.info("LiveKit integration: All patches applied successfully")
            
        except ImportError as e:
            logger.warning(f"LiveKit integration: Failed to import modules: {e}")
        except Exception as e:
            logger.error(f"LiveKit integration: Error during patching: {e}")
    
    def _try_patch_deepgram(self):
        """Try to patch Deepgram STT if available."""
        try:
            import livekit.plugins.deepgram
            from livekit.plugins.deepgram import stt as deepgram_stt
            
            # Patch Deepgram STT class
            self._patch_deepgram_stt(deepgram_stt)
            logger.info("LiveKit integration: Deepgram STT patches applied")
            
        except ImportError:
            logger.debug("LiveKit integration: Deepgram plugin not available, skipping patches")
        except Exception as e:
            logger.warning(f"LiveKit integration: Failed to patch Deepgram: {e}")
    
    def _try_patch_cartesia(self):
        """Try to patch Cartesia TTS if available."""
        try:
            import livekit.plugins.cartesia
            from livekit.plugins.cartesia import tts as cartesia_tts
            
            # Patch Cartesia TTS class
            self._patch_cartesia_tts(cartesia_tts)
            logger.info("LiveKit integration: Cartesia TTS patches applied")
            
        except ImportError:
            logger.debug("LiveKit integration: Cartesia plugin not available, skipping patches")
        except Exception as e:
            logger.warning(f"LiveKit integration: Failed to patch Cartesia: {e}")
    
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
                
                # Register metrics handler on the session
                if hasattr(self_session, 'on'):
                    @self_session.on("metrics_collected")
                    def _on_metrics_collected(event):
                        logger.debug(f"LiveKit: Metrics collected event received: {type(event.metrics).__name__}")
                        integration._capture_metrics_span(event, self_session)
                
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
                    self_session._ze_user_turn_transcripts = []  # Accumulate transcripts
            
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
                    
                    # Add accumulated transcripts to the user turn span
                    if hasattr(self_session, '_ze_user_turn_transcripts') and self_session._ze_user_turn_transcripts:
                        all_transcripts = ' '.join([t['text'] for t in self_session._ze_user_turn_transcripts])
                        turn_span.attributes["stt.transcript"] = all_transcripts
                        turn_span.attributes["stt.provider"] = "deepgram"
                        turn_span.attributes["stt.transcript_count"] = len(self_session._ze_user_turn_transcripts)
                        
                        # Add average confidence if available
                        confidences = [t['confidence'] for t in self_session._ze_user_turn_transcripts if t['confidence'] is not None]
                        if confidences:
                            turn_span.attributes["stt.avg_confidence"] = sum(confidences) / len(confidences)
                        
                        # Include all raw data from transcripts
                        all_raw_data = []
                        for t in self_session._ze_user_turn_transcripts:
                            if 'raw_data' in t:
                                all_raw_data.append(t['raw_data'])
                        if all_raw_data:
                            turn_span.attributes["stt.raw_data"] = json.dumps(all_raw_data)
                        
                        logger.info(f"Added {len(self_session._ze_user_turn_transcripts)} transcripts to user turn: {all_transcripts[:100]}...")
                    
                    # Store info about this turn for delayed transcript association
                    self_session._ze_last_user_turn_info = {
                        'span_id': turn_span.span_id,
                        'trace_id': turn_span.trace_id,
                        'speech_id': speech_id,
                        'end_time': time.time()
                    }
                    
                    if turn_span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(turn_span.span_id)
                        integration.tracer.end_span(turn_span)
                    
                    # Clean up
                    self_session._ze_user_turn_span = None
                    self_session._ze_user_turn_speech_id = None
                    if hasattr(self_session, '_ze_user_turn_transcripts'):
                        self_session._ze_user_turn_transcripts = []
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
                'on_final_transcript': AgentActivity.on_final_transcript,
            }
        
        integration = self
        
        # Patch on_final_transcript to track STT under user turns
        original_on_final_transcript = self._original_agent_activity['on_final_transcript']
        
        @wraps(original_on_final_transcript)
        def wrapped_on_final_transcript(self_activity, ev):
            """Wrap on_final_transcript to create STT spans under user turns."""
            # Get session from activity
            session = self_activity._session
            session_id = getattr(session, "_ze_session_id", None)
            
            # Check if using Deepgram
            is_deepgram = False
            if self_activity.stt and hasattr(self_activity.stt, '__class__'):
                stt_module = self_activity.stt.__class__.__module__
                is_deepgram = 'deepgram' in stt_module.lower()
            
            logger.debug(f"on_final_transcript called - session_id={session_id}, is_deepgram={is_deepgram}")
            
            if session_id and is_deepgram and ev.alternatives:
                transcript_text = ev.alternatives[0].text if ev.alternatives else ""
                
                # Extract raw event data first (before any conditionals)
                raw_event_data = {}
                try:
                    # Try to convert the event to a dict representation
                    if hasattr(ev, '__dict__'):
                        raw_event_data = {k: v for k, v in ev.__dict__.items() if not k.startswith('_')}
                    
                    # Add alternatives data
                    if ev.alternatives:
                        raw_event_data['alternatives'] = []
                        for alt in ev.alternatives:
                            alt_data = {}
                            if hasattr(alt, '__dict__'):
                                alt_data = {k: v for k, v in alt.__dict__.items() if not k.startswith('_')}
                            else:
                                # Try to extract common attributes
                                for attr in ['text', 'confidence', 'language', 'speaker_id', 'words', 'start_time', 'end_time']:
                                    if hasattr(alt, attr):
                                        alt_data[attr] = getattr(alt, attr)
                            raw_event_data['alternatives'].append(alt_data)
                    
                    # Add any other event attributes
                    for attr in ['type', 'is_final', 'request_id', 'created_at', 'duration', 'metadata']:
                        if hasattr(ev, attr):
                            raw_event_data[attr] = getattr(ev, attr)
                except Exception as e:
                    logger.debug(f"Failed to serialize Deepgram event data: {e}")
                    raw_event_data = {"error": f"Failed to serialize: {str(e)}"}
                
                # Try to get the active user turn span or the last closed one
                user_turn_span = getattr(session, "_ze_user_turn_span", None)
                last_user_turn_info = getattr(session, "_ze_last_user_turn_info", None)
                
                if user_turn_span:
                    # User is still speaking, add transcript to the active span
                    logger.debug(f"Adding transcript to active user turn: {transcript_text[:50]}...")
                    if hasattr(user_turn_span, 'attributes'):
                        user_turn_span.attributes["stt.transcript"] = transcript_text
                        user_turn_span.attributes["stt.provider"] = "deepgram"
                        if hasattr(ev.alternatives[0], 'confidence'):
                            user_turn_span.attributes["stt.confidence"] = ev.alternatives[0].confidence
                        if hasattr(ev.alternatives[0], 'language'):
                            user_turn_span.attributes["stt.language"] = ev.alternatives[0].language
                    
                    # Also accumulate transcripts
                    if hasattr(session, '_ze_user_turn_transcripts'):
                        session._ze_user_turn_transcripts.append({
                            'text': transcript_text,
                            'confidence': ev.alternatives[0].confidence if hasattr(ev.alternatives[0], 'confidence') else None,
                            'language': ev.alternatives[0].language if hasattr(ev.alternatives[0], 'language') else None,
                            'timestamp': time.time(),
                            'raw_data': raw_event_data  # Include raw data in accumulation
                        })
                    
                    # Create base attributes
                    span_attributes = {
                        "service.name": "deepgram",
                        "kind": "stt_result",
                        "stt.provider": "deepgram",
                        "stt.is_final": True,
                        "stt.transcript": transcript_text,
                        "stt.raw_data": json.dumps(raw_event_data),  # Still include raw data for completeness
                    }
                    
                    # Extract top-level event attributes
                    if 'type' in raw_event_data:
                        span_attributes["stt.type"] = raw_event_data['type']
                    if 'request_id' in raw_event_data:
                        span_attributes["stt.request_id"] = raw_event_data['request_id']
                    if 'recognition_usage' in raw_event_data:
                        span_attributes["stt.recognition_usage"] = raw_event_data['recognition_usage']
                    
                    # Extract first alternative's details
                    if raw_event_data.get('alternatives') and len(raw_event_data['alternatives']) > 0:
                        first_alt = raw_event_data['alternatives'][0]
                        if 'confidence' in first_alt:
                            span_attributes["stt.confidence"] = first_alt['confidence']
                        if 'language' in first_alt:
                            span_attributes["stt.language"] = first_alt['language']
                        if 'speaker_id' in first_alt:
                            span_attributes["stt.speaker_id"] = first_alt['speaker_id']
                        if 'start_time' in first_alt:
                            span_attributes["stt.start_time"] = first_alt['start_time']
                        if 'end_time' in first_alt:
                            span_attributes["stt.end_time"] = first_alt['end_time']
                        
                        # If there are words, include word count
                        if 'words' in first_alt and isinstance(first_alt['words'], list):
                            span_attributes["stt.word_count"] = len(first_alt['words'])
                    
                    # Also create a child span for the transcript
                    span = integration.tracer.start_span(
                        name="deepgram.stt.transcript",
                        attributes=span_attributes,
                        tags={"integration": "livekit", "deepgram": "true", "turn_type": "user"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                    )
                    
                    # Make it a child of the user turn
                    span.parent_id = user_turn_span.span_id
                    span.trace_id = user_turn_span.trace_id
                    
                    # End the span immediately
                    if span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(span.span_id)
                        integration.tracer.end_span(span)
                
                elif last_user_turn_info and time.time() - last_user_turn_info.get('end_time', 0) < 5.0:
                    # User turn recently ended, create a standalone transcript span linked to that turn
                    logger.debug(f"Creating transcript span for recently ended turn: {transcript_text[:50]}...")
                    
                    # Also accumulate for delayed transcripts
                    if hasattr(session, '_ze_user_turn_transcripts'):
                        session._ze_user_turn_transcripts.append({
                            'text': transcript_text,
                            'confidence': ev.alternatives[0].confidence if hasattr(ev.alternatives[0], 'confidence') else None,
                            'language': ev.alternatives[0].language if hasattr(ev.alternatives[0], 'language') else None,
                            'timestamp': time.time(),
                            'raw_data': raw_event_data  # Include raw data in accumulation
                        })
                    
                    # Create base attributes (same extraction as above)
                    span_attributes = {
                        "service.name": "deepgram",
                        "kind": "stt_result",
                        "stt.provider": "deepgram",
                        "stt.is_final": True,
                        "stt.transcript": transcript_text,
                        "stt.user_turn_id": last_user_turn_info.get('speech_id'),
                        "stt.raw_data": json.dumps(raw_event_data),  # Still include raw data for completeness
                    }
                    
                    # Extract top-level event attributes
                    if 'type' in raw_event_data:
                        span_attributes["stt.type"] = raw_event_data['type']
                    if 'request_id' in raw_event_data:
                        span_attributes["stt.request_id"] = raw_event_data['request_id']
                    if 'recognition_usage' in raw_event_data:
                        span_attributes["stt.recognition_usage"] = raw_event_data['recognition_usage']
                    
                    # Extract first alternative's details
                    if raw_event_data.get('alternatives') and len(raw_event_data['alternatives']) > 0:
                        first_alt = raw_event_data['alternatives'][0]
                        if 'confidence' in first_alt:
                            span_attributes["stt.confidence"] = first_alt['confidence']
                        if 'language' in first_alt:
                            span_attributes["stt.language"] = first_alt['language']
                        if 'speaker_id' in first_alt:
                            span_attributes["stt.speaker_id"] = first_alt['speaker_id']
                        if 'start_time' in first_alt:
                            span_attributes["stt.start_time"] = first_alt['start_time']
                        if 'end_time' in first_alt:
                            span_attributes["stt.end_time"] = first_alt['end_time']
                        
                        # If there are words, include word count
                        if 'words' in first_alt and isinstance(first_alt['words'], list):
                            span_attributes["stt.word_count"] = len(first_alt['words'])
                    
                    span = integration.tracer.start_span(
                        name="deepgram.stt.transcript",
                        attributes=span_attributes,
                        tags={"integration": "livekit", "deepgram": "true", "turn_type": "user", "delayed": "true"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                    )
                    
                    # Link to the last user turn's trace
                    if 'trace_id' in last_user_turn_info:
                        span.trace_id = last_user_turn_info['trace_id']
                        span.parent_id = last_user_turn_info.get('span_id')
                    
                    # End the span immediately
                    if span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(span.span_id)
                        integration.tracer.end_span(span)
                else:
                    logger.debug(f"No active or recent user turn found for transcript: {transcript_text[:50]}...")
            
            # Call original method
            return original_on_final_transcript(self_activity, ev)
        
        wrapped_on_final_transcript._ze_patched = True
        AgentActivity.on_final_transcript = wrapped_on_final_transcript
        logger.info("LiveKit integration: Patched AgentActivity.on_final_transcript for Deepgram tracking")
        
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
                logger.debug(f"LiveKit: Unknown metrics type: {metrics_type}")
                return
            
            logger.info(f"LiveKit: Creating metrics span {span_name} for {metrics_type}")
            
            # Get current turn span if available
            current_turn_span = getattr(session_instance, "_ze_current_turn_span", None)
            session_id = getattr(session_instance, "_ze_session_id", None)
            
            if not session_id:
                return
            
            # Create a span for this metric
            span = self.tracer.start_span(
                name=span_name,
                kind=metrics_type.lower().replace("metrics", ""),  # Set kind as parameter
                attributes={
                    "service.name": "livekit",
                },
                tags={"integration": "livekit", "metrics_type": metrics_type},
                session_id=session_id,
                session_name=getattr(session_instance, "_ze_session_name", None),
            )
            
            # If we have a current turn span, make this a child of it
            if current_turn_span:
                span.parent_id = current_turn_span.span_id
                span.trace_id = current_turn_span.trace_id
            
            # Add provider and model for LLM spans
            if metrics_type == "LLMMetrics":
                # Try to get model information from metrics or session
                model = getattr(metrics, "model", None)
                if not model and hasattr(session_instance, "_ze_session_tags"):
                    model = session_instance._ze_session_tags.get("llm")
                
                if model:
                    span.attributes["model"] = model
                # Set provider as livekit for now
                span.attributes["provider"] = "livekit"
            
            # Add metric-specific attributes
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
            attributes["output_tokens"] = metrics.completion_tokens
        if hasattr(metrics, "prompt_tokens"):
            attributes["input_tokens"] = metrics.prompt_tokens
        if hasattr(metrics, "total_tokens"):
            attributes["total_tokens"] = metrics.total_tokens
        if hasattr(metrics, "ttft"):
            attributes["ttft_ms"] = metrics.ttft * 1000  # Convert to milliseconds
            span.set_signal("ttft_ms", metrics.ttft * 1000)
        if hasattr(metrics, "tokens_per_second"):
            attributes["throughput"] = metrics.tokens_per_second
        
        # TTS Metrics
        if hasattr(metrics, "characters_count"):
            attributes["characters_count"] = metrics.characters_count  # Changed from tts.characters_count
        if hasattr(metrics, "ttfb"):
            attributes["ttfb_ms"] = metrics.ttfb * 1000  # Changed from tts.ttfb and converted to ms
            span.set_signal("ttfb_ms", metrics.ttfb * 1000)
        
        # Add standard TTS fields for metrics spans
        if metrics_type == "TTSMetrics":
            # These are required fields for tts_span_metrics
            attributes["provider"] = "unknown"  # We don't know the provider from metrics alone
            attributes["streaming"] = True  # LiveKit TTS is typically streaming
            
            # Try to infer provider from the activity if possible
            # This is a best effort - the direct Cartesia integration provides more accurate data
        
        # EOU Metrics (End of Utterance)
        if hasattr(metrics, "end_of_utterance_delay"):
            attributes["eou.end_of_utterance_delay"] = metrics.end_of_utterance_delay
            span.set_signal("eou_delay_ms", metrics.end_of_utterance_delay * 1000)
        if hasattr(metrics, "transcription_delay"):
            attributes["eou.transcription_delay"] = metrics.transcription_delay
            span.set_signal("transcription_delay_ms", metrics.transcription_delay * 1000)
        if hasattr(metrics, "on_user_turn_completed_delay"):
            attributes["eou.on_user_turn_completed_delay"] = metrics.on_user_turn_completed_delay
        if hasattr(metrics, "last_speaking_time"):
            attributes["eou.last_speaking_time"] = metrics.last_speaking_time
        if hasattr(metrics, "timestamp"):
            attributes["eou.timestamp"] = metrics.timestamp
        
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
                if self._original_agent_activity['on_final_transcript']:
                    AgentActivity.on_final_transcript = self._original_agent_activity['on_final_transcript']
            except Exception:
                pass
        
        # Restore function_tool
        if self._original_function_tool:
            try:
                import livekit.agents.llm as llm_module
                llm_module.function_tool = self._original_function_tool
            except Exception:
                pass
    
    def _patch_deepgram_stt(self, deepgram_stt_module) -> None:
        """Patch Deepgram STT for observability."""
        try:
            STT = deepgram_stt_module.STT
            SpeechStream = deepgram_stt_module.SpeechStream
            
            # Check if already patched
            if hasattr(STT.stream, '_ze_patched') and STT.stream._ze_patched:
                logger.debug("LiveKit integration: Deepgram STT already patched, skipping")
                return
            
            # Store originals
            if not self._original_deepgram_stt:
                self._original_deepgram_stt = {
                    '_recognize_impl': STT._recognize_impl,
                    'stream': STT.stream,
                }
                self._original_deepgram_stream = {
                    '__init__': SpeechStream.__init__,
                    '_process_stream_event': SpeechStream._process_stream_event,
                    'aclose': SpeechStream.aclose,
                }
            
            integration = self
            
            # Patch non-streaming recognize
            @wraps(self._original_deepgram_stt['_recognize_impl'])
            async def wrapped_recognize_impl(self_stt, buffer, **kwargs):
                """Wrap Deepgram's non-streaming recognize method."""
                # Try to find session context by walking the call stack
                session = integration._find_session_from_context()
                session_id = getattr(session, "_ze_session_id", None) if session else None
                current_turn_span = getattr(session, "_ze_current_turn_span", None) if session else None
                
                if session_id:
                    logger.debug(f"Deepgram: Creating recognize span for session {session_id}")
                    span = integration.tracer.start_span(
                        name="deepgram.stt.recognize",
                        attributes={
                            "service.name": "deepgram",
                            "kind": "stt",
                            "stt.provider": "deepgram",
                            "stt.streaming": False,
                            "stt.model": str(self_stt._opts.model),
                            "stt.language": str(self_stt._opts.language) if self_stt._opts.language else None,
                            "stt.detect_language": self_stt._opts.detect_language,
                            "stt.interim_results": self_stt._opts.interim_results,
                            "stt.smart_format": self_stt._opts.smart_format,
                            "stt.sample_rate": self_stt._opts.sample_rate,
                            "stt.buffer_size": len(buffer) if hasattr(buffer, '__len__') else None,
                        },
                        tags={"integration": "livekit", "deepgram": "true"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                    )
                    
                    # Make it a child of current turn if available
                    if current_turn_span:
                        span.parent_id = current_turn_span.span_id
                        span.trace_id = current_turn_span.trace_id
                    
                    start_time = time.time()
                    
                    try:
                        result = await integration._original_deepgram_stt['_recognize_impl'](
                            self_stt, buffer, **kwargs
                        )
                        
                        # Add result metadata
                        if result:
                            # Serialize the entire result data
                            raw_result_data = {}
                            try:
                                # Try to convert the result to a dict representation
                                if hasattr(result, '__dict__'):
                                    raw_result_data = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
                                
                                # Add alternatives data
                                if hasattr(result, 'alternatives') and result.alternatives:
                                    raw_result_data['alternatives'] = []
                                    for alt in result.alternatives:
                                        alt_data = {}
                                        if hasattr(alt, '__dict__'):
                                            alt_data = {k: v for k, v in alt.__dict__.items() if not k.startswith('_')}
                                        else:
                                            # Try to extract common attributes
                                            for attr in ['text', 'confidence', 'language', 'words', 'start_time', 'end_time']:
                                                if hasattr(alt, attr):
                                                    alt_data[attr] = getattr(alt, attr)
                                        raw_result_data['alternatives'].append(alt_data)
                                
                                # Add any other result attributes
                                for attr in ['type', 'is_final', 'request_id', 'created_at', 'duration', 'metadata', 'channel']:
                                    if hasattr(result, attr):
                                        raw_result_data[attr] = getattr(result, attr)
                                
                                # Store the raw data
                                span.attributes["stt.raw_data"] = json.dumps(raw_result_data)
                            except Exception as e:
                                logger.debug(f"Failed to serialize Deepgram result data: {e}")
                                span.attributes["stt.raw_data"] = json.dumps({"error": f"Failed to serialize: {str(e)}"})
                            
                            # Also add specific attributes for quick access
                            if hasattr(result, 'alternatives') and result.alternatives:
                                first_alt = result.alternatives[0]
                                if hasattr(first_alt, 'text'):
                                    span.attributes["stt.transcript"] = first_alt.text  # Include full transcript
                                if hasattr(first_alt, 'confidence'):
                                    span.attributes["stt.confidence"] = first_alt.confidence
                                if hasattr(first_alt, 'language'):
                                    span.attributes["stt.detected_language"] = first_alt.language
                        
                        span.attributes["stt.duration_ms"] = int((time.time() - start_time) * 1000)
                        
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
                    # No session context, just call original
                    return await integration._original_deepgram_stt['_recognize_impl'](
                        self_stt, buffer, **kwargs
                    )
            
            # Mark as patched
            wrapped_recognize_impl._ze_patched = True
            
            # Patch streaming
            @wraps(self._original_deepgram_stt['stream'])
            def wrapped_stream(self_stt, **kwargs):
                """Wrap Deepgram's stream method."""
                # Create the original stream
                stream = integration._original_deepgram_stt['stream'](self_stt, **kwargs)
                
                # Try to find session context
                session = integration._find_session_from_context()
                if session:
                    # Store session reference for the stream
                    integration._stream_sessions[id(stream)] = session
                    
                    # Create span for stream creation
                    session_id = getattr(session, "_ze_session_id", None)
                    current_turn_span = getattr(session, "_ze_current_turn_span", None)
                    
                    if session_id:
                        logger.debug(f"Deepgram: Creating stream span for session {session_id}")
                        span = integration.tracer.start_span(
                            name="deepgram.stt.stream_start",
                            attributes={
                                "service.name": "deepgram",
                                "kind": "stt",
                                "stt.provider": "deepgram",
                                "stt.streaming": True,
                                "stt.model": str(self_stt._opts.model),
                                "stt.language": str(self_stt._opts.language) if self_stt._opts.language else None,
                                "stt.detect_language": self_stt._opts.detect_language,
                                "stt.interim_results": self_stt._opts.interim_results,
                                "stt.smart_format": self_stt._opts.smart_format,
                                "stt.sample_rate": self_stt._opts.sample_rate,
                            },
                            tags={"integration": "livekit", "deepgram": "true"},
                            session_id=session_id,
                            session_name=getattr(session, "_ze_session_name", None),
                        )
                        
                        # Make it a child of current turn if available
                        if current_turn_span:
                            span.parent_id = current_turn_span.span_id
                            span.trace_id = current_turn_span.trace_id
                        
                        # Store span ID for the stream
                        stream._ze_stream_span_id = span.span_id
                        stream._ze_start_time = time.time()
                        
                        # End the span immediately (stream lifecycle tracked separately)
                        if span.span_id not in integration._ended_span_ids:
                            integration._ended_span_ids.add(span.span_id)
                            integration.tracer.end_span(span)
                
                return stream
            
            # Mark as patched
            wrapped_stream._ze_patched = True
            
            # Apply patches
            STT._recognize_impl = wrapped_recognize_impl
            STT.stream = wrapped_stream
            
            # Patch SpeechStream methods
            if not hasattr(SpeechStream._process_stream_event, '_ze_patched') or not SpeechStream._process_stream_event._ze_patched:
                self._patch_deepgram_stream_process(SpeechStream)
            if not hasattr(SpeechStream.aclose, '_ze_patched') or not SpeechStream.aclose._ze_patched:
                self._patch_deepgram_stream_close(SpeechStream)
            
            logger.debug("LiveKit integration: Deepgram patches applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to patch Deepgram STT: {e}")
    
    def _patch_cartesia_tts(self, cartesia_tts_module) -> None:
        """Patch Cartesia TTS for observability."""
        try:
            TTS = cartesia_tts_module.TTS
            SynthesizeStream = cartesia_tts_module.SynthesizeStream
            
            # Import the default value
            try:
                from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
            except ImportError:
                DEFAULT_API_CONNECT_OPTIONS = None
            
            # Check if already patched
            if hasattr(TTS.stream, '_ze_patched') and TTS.stream._ze_patched:
                logger.debug("LiveKit integration: Cartesia TTS already patched, skipping")
                return
            
            # Store originals - ensure we get the unbound methods
            try:
                # Try to get unbound method if it's a bound method
                stream_method = TTS.stream.__func__ if hasattr(TTS.stream, '__func__') else TTS.stream
                run_method = SynthesizeStream._run.__func__ if hasattr(SynthesizeStream._run, '__func__') else SynthesizeStream._run
                init_method = TTS.__init__.__func__ if hasattr(TTS.__init__, '__func__') else TTS.__init__
            except AttributeError:
                # Fallback to direct assignment if __func__ access fails
                stream_method = TTS.stream
                run_method = SynthesizeStream._run
                init_method = TTS.__init__
                
            self._original_cartesia_tts = {
                'stream': stream_method,
                '__init__': init_method,
            }
            self._original_cartesia_stream = {
                '_run': run_method,
            }
            
            integration = self
            
            # Patch TTS __init__ to store activity reference
            @wraps(self._original_cartesia_tts['__init__'])
            def wrapped_init(self_tts, *args, **kwargs):
                """Wrap TTS init to capture context."""
                # Call original init
                original_init = integration._original_cartesia_tts['__init__']
                if hasattr(original_init, '__func__'):
                    result = original_init(self_tts, *args, **kwargs)
                else:
                    result = original_init(self_tts, *args, **kwargs)
                
                # Try to capture the current session/activity context
                session = integration._find_session_from_context()
                if session:
                    self_tts._ze_session = session
                    logger.debug(f"Cartesia TTS: Captured session during init")
                
                return result
            
            # Mark as patched
            wrapped_init._ze_patched = True
            
            # Patch stream creation
            @wraps(self._original_cartesia_tts['stream'])
            def wrapped_stream(self_tts, *, conn_options=None):
                """Wrap Cartesia's stream method."""
                # Create the original stream without any modification
                if conn_options is None and DEFAULT_API_CONNECT_OPTIONS is not None:
                    conn_options = DEFAULT_API_CONNECT_OPTIONS
                
                # Call the original method
                original_stream = integration._original_cartesia_tts['stream']
                if hasattr(original_stream, '__func__'):
                    stream = original_stream(self_tts, conn_options=conn_options)
                else:
                    stream = original_stream(self_tts, conn_options=conn_options)
                
                # Only try to store session reference for later use, don't modify the stream
                try:
                    session = integration._find_session_from_context()
                    if not session and hasattr(self_tts, '_ze_session'):
                        session = self_tts._ze_session
                    
                    if session:
                        # Store session reference for the stream (weakref to avoid memory leaks)
                        integration._stream_sessions[id(stream)] = session
                        
                        # Add a minimal wrapper to track input text
                        original_push_text = stream.push_text
                        stream._ze_input_texts = []
                        
                        def track_push_text(text):
                            # Store text for later use in span
                            stream._ze_input_texts.append(text)
                            return original_push_text(text)
                        
                        stream.push_text = track_push_text
                        
                        # Create a simple span for stream creation
                        session_id = getattr(session, "_ze_session_id", None)
                        if session_id:
                            span = integration.tracer.start_span(
                                name="cartesia.tts.stream_start",
                                kind="tts",  # Set kind as parameter, not attribute
                                attributes={
                                    "service.name": "cartesia",
                                    "provider": "cartesia",
                                    "streaming": True,
                                },
                                tags={"integration": "livekit", "cartesia": "true"},
                                session_id=session_id,
                                session_name=getattr(session, "_ze_session_name", None),
                            )
                            
                            # Add basic TTS options if available
                            if hasattr(self_tts, '_opts'):
                                opts = self_tts._opts
                                span.attributes.update({
                                    "voice_id": str(opts.voice) if hasattr(opts, 'voice') else None,
                                    "model_id": str(opts.model) if hasattr(opts, 'model') else None,
                                    "language_code": str(opts.language) if hasattr(opts, 'language') else None,
                                    "output_format": str(opts.encoding) if hasattr(opts, 'encoding') else None,
                                })
                            
                            # Link to current turn if available
                            current_turn_span = getattr(session, "_ze_current_turn_span", None)
                            if current_turn_span:
                                span.parent_id = current_turn_span.span_id
                                span.trace_id = current_turn_span.trace_id
                            
                            # End immediately
                            if span.span_id not in integration._ended_span_ids:
                                integration._ended_span_ids.add(span.span_id)
                                integration.tracer.end_span(span)
                except Exception as e:
                    logger.debug(f"Cartesia: Failed to track stream creation: {e}")
                
                return stream
            
            # Mark as patched
            wrapped_stream._ze_patched = True
            
            # Patch stream processing
            @wraps(self._original_cartesia_stream['_run'])
            async def wrapped_run(self_stream, output_emitter):
                """Wrap Cartesia's stream _run method."""
                # Get session from stream if available
                session = getattr(self_stream, '_ze_session', None)
                if not session and id(self_stream) in integration._stream_sessions:
                    session = integration._stream_sessions[id(self_stream)]
                
                logger.debug(f"Cartesia _run: Found session from stream: {session is not None}")
                
                session_id = getattr(session, "_ze_session_id", None) if session else None
                current_turn_span = getattr(session, "_ze_current_turn_span", None) if session else None
                
                logger.debug(f"Cartesia _run: Session ID: {session_id}, Current turn span: {current_turn_span is not None}")
                
                if session_id:
                    logger.debug(f"Cartesia: Creating synthesis span for session {session_id}")
                    
                    # Get TTS options from the stream
                    tts_opts = getattr(self_stream, '_opts', None)
                    
                    span = integration.tracer.start_span(
                        name="cartesia.tts.synthesize",
                        kind="tts",  # Set kind as parameter, not attribute
                        attributes={
                            "service.name": "cartesia",
                            "provider": "cartesia",  # Required by tts_span_metrics
                            "streaming": True,       # Required by tts_span_metrics
                        },
                        tags={"integration": "livekit", "cartesia": "true"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                    )
                    
                    # Add TTS options if available - mapped to tts_span_metrics fields
                    if tts_opts:
                        # Core fields for tts_span_metrics
                        span.attributes.update({
                            "voice_id": str(tts_opts.voice) if hasattr(tts_opts, 'voice') else None,
                            "model_id": str(tts_opts.model) if hasattr(tts_opts, 'model') else None,
                            "language_code": str(tts_opts.language) if hasattr(tts_opts, 'language') else None,
                            "output_format": str(tts_opts.encoding) if hasattr(tts_opts, 'encoding') else None,
                        })
                        
                        # Provider-specific data as JSON
                        provider_request_data = {}
                        if hasattr(tts_opts, 'sample_rate'):
                            provider_request_data['sample_rate'] = tts_opts.sample_rate
                        if hasattr(tts_opts, 'word_timestamps'):
                            provider_request_data['word_timestamps'] = tts_opts.word_timestamps
                        if hasattr(tts_opts, 'speed'):
                            provider_request_data['speed'] = tts_opts.speed
                        if hasattr(tts_opts, 'emotion'):
                            provider_request_data['emotion'] = str(tts_opts.emotion)
                        
                        if provider_request_data:
                            span.attributes['provider_request_data'] = provider_request_data
                    
                    # Make it a child of current turn if available
                    if current_turn_span:
                        span.parent_id = current_turn_span.span_id
                        span.trace_id = current_turn_span.trace_id
                    
                    start_time = time.time()
                    
                    try:
                        # Run the original method WITHOUT any wrapping or monitoring
                        original_run = integration._original_cartesia_stream['_run']
                        if hasattr(original_run, '__func__'):
                            # It's a bound method, we stored __func__
                            result = await original_run(self_stream, output_emitter)
                        else:
                            # It's already unbound or a function
                            result = await original_run(self_stream, output_emitter)
                        
                        # Add basic metadata after completion
                        span.attributes.update({
                            "audio_duration_ms": int((time.time() - start_time) * 1000),
                        })
                        
                        # Set input_data if we captured any text
                        if hasattr(self_stream, '_ze_input_texts') and self_stream._ze_input_texts:
                            input_text = ''.join(self_stream._ze_input_texts)
                            span.set_io(input_data=input_text)
                            # Also add characters_count for the metrics table
                            span.attributes["characters_count"] = len(input_text)
                        
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
                    # No session context, just call original
                    original_run = integration._original_cartesia_stream['_run']
                    if hasattr(original_run, '__func__'):
                        return await original_run(self_stream, output_emitter)
                    else:
                        return await original_run(self_stream, output_emitter)
            
            # Mark as patched
            wrapped_run._ze_patched = True
            
            # Apply patches
            TTS.__init__ = wrapped_init
            TTS.stream = wrapped_stream
            SynthesizeStream._run = wrapped_run
            
            logger.debug("LiveKit integration: Cartesia patches applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to patch Cartesia TTS: {e}")
    
    def _patch_deepgram_stream_process(self, SpeechStream) -> None:
        """Patch Deepgram SpeechStream._process_stream_event."""
        original_process_stream_event = SpeechStream._process_stream_event
        integration = self
        
        @wraps(original_process_stream_event)
        def wrapped_process_stream_event(self_stream, data):
            """Wrap stream event processing to capture results."""
            # Call original
            result = original_process_stream_event(self_stream, data)
            
            # Try to find session from our tracking
            session = integration._stream_sessions.get(id(self_stream))
            if session and hasattr(self_stream, '_ze_stream_span_id'):
                session_id = getattr(session, "_ze_session_id", None)
                current_turn_span = getattr(session, "_ze_current_turn_span", None)
                
                # Also check various possible event types for Deepgram
                event_type = data.get('type', '')
                is_transcript_event = event_type in ['Results', 'Result', 'transcript', 'transcription']
                
                # Commented out: Now handled in on_final_transcript to properly associate with user turns
                # if session_id and (is_transcript_event or 'alternatives' in data.get('channel', {})):
                #     # Only create spans for final transcripts
                #     is_final = data.get('is_final', False)
                #     
                #     if is_final:
                #         ... span creation code ...
            
            return result
        
        wrapped_process_stream_event._ze_patched = True
        SpeechStream._process_stream_event = wrapped_process_stream_event
    
    def _patch_deepgram_stream_close(self, SpeechStream) -> None:
        """Patch Deepgram SpeechStream.aclose."""
        original_aclose = SpeechStream.aclose
        integration = self
        
        @wraps(original_aclose)
        async def wrapped_aclose(self_stream):
            """Wrap stream close to track stream lifecycle."""
            # Find session and create end span
            session = integration._stream_sessions.get(id(self_stream))
            if session and hasattr(self_stream, '_ze_stream_span_id'):
                session_id = getattr(session, "_ze_session_id", None)
                current_turn_span = getattr(session, "_ze_current_turn_span", None)
                
                if session_id:
                    duration_ms = int((time.time() - self_stream._ze_start_time) * 1000)
                    
                    span = integration.tracer.start_span(
                        name="deepgram.stt.stream_end",
                        attributes={
                            "service.name": "deepgram",
                            "kind": "stt",
                            "stt.provider": "deepgram",
                            "stt.streaming": True,
                            "stt.stream_id": self_stream._ze_stream_span_id,
                            "stt.duration_ms": duration_ms,
                        },
                        tags={"integration": "livekit", "deepgram": "true"},
                        session_id=session_id,
                        session_name=getattr(session, "_ze_session_name", None),
                    )
                    
                    # Make it a child of current turn if available
                    if current_turn_span:
                        span.parent_id = current_turn_span.span_id
                        span.trace_id = current_turn_span.trace_id
                    
                    # End span immediately
                    if span.span_id not in integration._ended_span_ids:
                        integration._ended_span_ids.add(span.span_id)
                        integration.tracer.end_span(span)
                
                # Clean up tracking
                integration._stream_sessions.pop(id(self_stream), None)
            
            # Call original
            return await original_aclose(self_stream)
        
        wrapped_aclose._ze_patched = True
        SpeechStream.aclose = wrapped_aclose
    
    def _patch_agent_stt_node(self) -> None:
        """Patch Agent's stt_node to intercept STT stream creation with session context."""
        try:
            from livekit.agents.voice import Agent
            
            # Store original if not already done
            if not hasattr(Agent.default.stt_node, '_ze_original'):
                Agent.default.stt_node._ze_original = Agent.default.stt_node
            
            integration = self
            original_stt_node = Agent.default.stt_node._ze_original
            
            @wraps(original_stt_node)
            async def wrapped_stt_node(agent, audio, model_settings):
                """Wrapped stt_node that tracks Deepgram usage."""
                activity = agent._get_activity_or_raise()
                session = activity.session if hasattr(activity, 'session') else activity._session
                
                # Check if using Deepgram
                is_deepgram = False
                if activity.stt and hasattr(activity.stt, '__class__'):
                    stt_class_name = activity.stt.__class__.__name__
                    stt_module = activity.stt.__class__.__module__
                    is_deepgram = 'deepgram' in stt_module.lower()
                
                if is_deepgram and session and hasattr(session, '_ze_session_id'):
                    # Store session in a context var for Deepgram to find
                    import contextvars
                    if not hasattr(integration, '_session_context'):
                        integration._session_context = contextvars.ContextVar('ze_session')
                    
                    token = integration._session_context.set(session)
                    try:
                        # Create a span for STT streaming session
                        session_id = session._ze_session_id
                        current_turn_span = getattr(session, "_ze_current_turn_span", None)
                        
                        span = integration.tracer.start_span(
                            name="deepgram.stt.streaming_session",
                            attributes={
                                "service.name": "deepgram",
                                "kind": "stt",
                                "stt.provider": "deepgram",
                                "stt.streaming": True,
                            },
                            tags={"integration": "livekit", "deepgram": "true"},
                            session_id=session_id,
                            session_name=getattr(session, "_ze_session_name", None),
                        )
                        
                        # Make it a child of current turn if available
                        if current_turn_span:
                            span.parent_id = current_turn_span.span_id
                            span.trace_id = current_turn_span.trace_id
                        
                        start_time = time.time()
                        
                        try:
                            # Call original and yield results
                            async for event in original_stt_node(agent, audio, model_settings):
                                yield event
                        except asyncio.CancelledError:
                            # This is expected when the stream is cancelled
                            logger.debug("STT streaming session cancelled")
                            raise
                        finally:
                            # End the session span
                            span.attributes["stt.session_duration_ms"] = int((time.time() - start_time) * 1000)
                            if span.span_id not in integration._ended_span_ids:
                                integration._ended_span_ids.add(span.span_id)
                                integration.tracer.end_span(span)
                    finally:
                        integration._session_context.reset(token)
                else:
                    # Not Deepgram or no session, just pass through
                    async for event in original_stt_node(agent, audio, model_settings):
                        yield event
            
            # Apply the patch
            Agent.default.stt_node = staticmethod(wrapped_stt_node)
            logger.info("LiveKit integration: Agent.default.stt_node patched for Deepgram tracking")
            
        except Exception as e:
            logger.warning(f"Failed to patch Agent.stt_node: {e}")
    
    def _find_session_from_context(self):
        """Try to find the current session from the context."""
        # First try contextvars if available
        if hasattr(self, '_session_context'):
            try:
                session = self._session_context.get()
                if session:
                    return session
            except LookupError:
                pass
        
        # Fall back to stack inspection
        # This is a heuristic approach - we try to find the session
        # by looking at the current async task context
        try:
            import inspect
            
            # Walk up the call stack looking for a session
            for frame_info in inspect.stack():
                frame_locals = frame_info.frame.f_locals
                
                # Look for session in various places
                if 'session' in frame_locals and hasattr(frame_locals['session'], '_ze_session_id'):
                    return frame_locals['session']
                if 'self' in frame_locals:
                    self_obj = frame_locals['self']
                    if hasattr(self_obj, 'session') and hasattr(self_obj.session, '_ze_session_id'):
                        return self_obj.session
                    if hasattr(self_obj, '_session') and hasattr(self_obj._session, '_ze_session_id'):
                        return self_obj._session
                
                # Check for activity objects (common in voice agents)
                if 'activity' in frame_locals and hasattr(frame_locals['activity'], 'session'):
                    if hasattr(frame_locals['activity'].session, '_ze_session_id'):
                        return frame_locals['activity'].session
                
                # Check for context objects
                if 'ctx' in frame_locals and hasattr(frame_locals['ctx'], 'session'):
                    if hasattr(frame_locals['ctx'].session, '_ze_session_id'):
                        return frame_locals['ctx'].session
                
                # Check for self_tts which might have a reference to activity/session
                if 'self_tts' in frame_locals and hasattr(frame_locals['self_tts'], '_activity'):
                    if hasattr(frame_locals['self_tts']._activity, 'session'):
                        if hasattr(frame_locals['self_tts']._activity.session, '_ze_session_id'):
                            return frame_locals['self_tts']._activity.session
                
                # Check for wrapped_tts in voice agent contexts
                if 'wrapped_tts' in frame_locals:
                    wrapped = frame_locals['wrapped_tts']
                    # Check if it has a stored session
                    if hasattr(wrapped, '_ze_session'):
                        return wrapped._ze_session
                    # Check if it's part of an activity
                    if hasattr(wrapped, 'activity') and hasattr(wrapped.activity, 'session'):
                        if hasattr(wrapped.activity.session, '_ze_session_id'):
                            return wrapped.activity.session
        except Exception as e:
            logger.debug(f"Failed to find session from context: {e}")
        
        return None
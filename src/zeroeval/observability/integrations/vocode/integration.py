import asyncio
import logging
import time
import traceback
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from ..base import Integration
from .streaming_tracker import track_tts_streaming

logger = logging.getLogger(__name__)

# Context variable to store the conversation root span
_conversation_spans: ContextVar[dict[str, Any]] = ContextVar('vocode_conversation_spans', default={})


class VocodeIntegration(Integration):
    """
    Integration for Vocode's Python voice SDK.
    
    Patches Vocode to automatically create spans for:
    - Text-to-Speech synthesis (all providers)
    - Speech-to-Text transcription (all providers)
    - Conversation flow and orchestration
    - Voice pipeline latencies
    
    All spans within a conversation share the same trace ID.
    """
    
    PACKAGE_NAME = "vocode"

    def setup(self) -> None:
        """Set up Vocode integration by patching core methods."""
        try:
            # Import vocode modules
            import vocode.streaming.streaming_conversation as conv_module
            import vocode.streaming.synthesizer.base_synthesizer as synth_module
            import vocode.streaming.transcriber.base_transcriber as trans_module
            
            # Patch conversation lifecycle to create root span
            # This ensures all conversation activity is under one trace
            if hasattr(conv_module, 'StreamingConversation'):
                self._patch_method(
                    conv_module.StreamingConversation,
                    "start",
                    self._wrap_conversation_start
                )
                
                self._patch_method(
                    conv_module.StreamingConversation,
                    "terminate",
                    self._wrap_conversation_terminate
                )
                logger.debug("✅ Patched StreamingConversation.start and .terminate")
            
            # Patch TTS: BaseSynthesizer.create_speech
            # This is the universal entry point for ALL synthesizers
            if hasattr(synth_module, 'BaseSynthesizer'):
                self._patch_method(
                    synth_module.BaseSynthesizer,
                    "create_speech",
                    self._wrap_create_speech
                )
                logger.debug("✅ Patched BaseSynthesizer.create_speech")
            
            # Patch STT: We need to patch produce_nonblocking where it's defined
            patched_stt = False
            
            # In the installed version, produce_nonblocking is in AsyncWorker
            from vocode.streaming.utils import worker as worker_module
            
            # Patch AsyncWorker.produce_nonblocking (this covers all transcribers)
            if hasattr(worker_module, 'AsyncWorker'):
                try:
                    if hasattr(worker_module.AsyncWorker, 'produce_nonblocking'):
                        original = getattr(worker_module.AsyncWorker, 'produce_nonblocking')
                        wrapped = self._wrap_produce_nonblocking(original)
                        setattr(worker_module.AsyncWorker, 'produce_nonblocking', wrapped)
                        logger.debug("✅ Patched AsyncWorker.produce_nonblocking")
                        patched_stt = True
                except Exception as e:
                    logger.debug(f"Failed to patch AsyncWorker: {e}")
            
            # Also patch ThreadAsyncWorker.produce_nonblocking if it exists
            if hasattr(worker_module, 'ThreadAsyncWorker'):
                try:
                    if hasattr(worker_module.ThreadAsyncWorker, 'produce_nonblocking'):
                        original = getattr(worker_module.ThreadAsyncWorker, 'produce_nonblocking')
                        wrapped = self._wrap_produce_nonblocking(original)  
                        setattr(worker_module.ThreadAsyncWorker, 'produce_nonblocking', wrapped)
                        logger.debug("✅ Patched ThreadAsyncWorker.produce_nonblocking")
                        patched_stt = True
                except Exception as e:
                    logger.debug(f"Failed to patch ThreadAsyncWorker: {e}")
            
            if not patched_stt:
                logger.warning("⚠️ Could not patch any STT transcriber classes")
            
            logger.info("✅ Vocode integration setup complete")
            
        except ImportError as e:
            logger.debug(f"Vocode not installed, skipping integration: {e}")
        except Exception as e:
            logger.error(f"Failed to setup Vocode integration: {e}")
            raise
    
    def _wrap_conversation_start(self, original: Callable) -> Callable:
        """
        Wrap StreamingConversation.start to create a root span for the entire conversation.
        This span will be the parent of all TTS, STT, and LLM spans.
        """
        tracer = self.tracer
        
        @wraps(original)
        async def async_wrapper(conversation_instance, *args, **kwargs):
            """Start a conversation and create a root span."""
            # Get conversation ID
            conversation_id = getattr(conversation_instance, 'id', 'unknown')
            
            # Create root span for the conversation
            # This span will stay active for the entire conversation
            attributes = {
                "service.name": "vocode",
                "vocode.conversation.id": conversation_id,
                "vocode.conversation.type": "streaming",
            }
            
            # Add agent info if available
            if hasattr(conversation_instance, 'agent'):
                agent = conversation_instance.agent
                agent_class = agent.__class__.__name__
                attributes["vocode.agent.type"] = agent_class
                if hasattr(agent, 'agent_config'):
                    config = agent.agent_config
                    if hasattr(config, 'model_name'):
                        attributes["vocode.agent.model"] = config.model_name
            
            # Add transcriber info if available
            if hasattr(conversation_instance, 'transcriber'):
                transcriber = conversation_instance.transcriber
                transcriber_class = transcriber.__class__.__name__
                attributes["vocode.transcriber.type"] = transcriber_class
            
            # Add synthesizer info if available
            if hasattr(conversation_instance, 'synthesizer'):
                synthesizer = conversation_instance.synthesizer
                synthesizer_class = synthesizer.__class__.__name__
                attributes["vocode.synthesizer.type"] = synthesizer_class
            
            # Create the root span - this will be a new trace
            root_span = tracer.start_span(
                name="vocode.conversation",
                attributes=attributes,
                is_new_trace=True  # Start a new trace for this conversation
            )
            
            # Store the root span in context for this conversation
            conversation_spans = _conversation_spans.get()
            conversation_spans[conversation_id] = root_span
            _conversation_spans.set(conversation_spans)
            
            logger.info(f"[Vocode] Starting conversation {conversation_id} with trace {root_span.trace_id}")
            
            try:
                # Additionally, wrap the transcriber's output queue to capture STT results
                try:
                    if hasattr(conversation_instance, 'transcriber'):
                        transcriber = conversation_instance.transcriber
                        # Only wrap once per queue instance
                        queue_obj = getattr(transcriber, 'output_queue', None)
                        if queue_obj is not None and not getattr(queue_obj, '_ze_stt_wrapped', False):
                            self._wrap_transcriber_output_queue(transcriber, conversation_id)
                            setattr(queue_obj, '_ze_stt_wrapped', True)
                            logger.debug("✅ Wrapped transcriber.output_queue.put_nowait for STT tracing")
                except Exception as wrap_err:
                    logger.debug(f"Failed to wrap transcriber output queue for STT: {wrap_err}")

                # Call the original start method
                result = await original(conversation_instance, *args, **kwargs)
                return result
            except Exception as exc:
                # Record exception in root span
                root_span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=traceback.format_exc()
                )
                # Don't end the span here - let terminate do it
                raise
        
        return async_wrapper

    def _wrap_transcriber_output_queue(self, transcriber_instance, conversation_id: str) -> None:
        """
        Wrap the transcriber's output_queue.put_nowait to create STT spans for every
        Transcription emitted by providers that write directly to the queue.
        """
        import types
        tracer = self.tracer

        queue_obj = getattr(transcriber_instance, 'output_queue', None)
        if queue_obj is None:
            return

        # Keep original method
        original_put = getattr(queue_obj, 'put_nowait')

        def put_nowait_wrapper(self_queue, item):
            try:
                # Only trace Transcription-like items; avoid duplicates
                if item is not None and hasattr(item, 'message') and not getattr(item, '_ze_stt_traced', False):
                    class_name = transcriber_instance.__class__.__name__
                    provider = class_name.replace('Transcriber', '').lower()

                    message = getattr(item, 'message', '') or ''
                    is_final = bool(getattr(item, 'is_final', False))
                    confidence = float(getattr(item, 'confidence', 0.0) or 0.0)
                    is_interrupt = bool(getattr(item, 'is_interrupt', False))
                    duration_seconds = getattr(item, 'duration_seconds', None)
                    wpm_value = None
                    if hasattr(item, 'wpm') and callable(item.wpm):
                        try:
                            wpm_value = item.wpm()
                        except Exception:
                            wpm_value = None

                    attributes = {
                        'service.name': 'vocode',
                        'provider': provider,
                        'vocode.stt.provider': provider,
                        'vocode.stt.text': message,
                        'vocode.stt.text_length': len(message),
                        'vocode.stt.is_final': is_final,
                        'vocode.stt.confidence': confidence,
                        'vocode.stt.is_interrupt': is_interrupt,
                    }
                    if conversation_id:
                        attributes['vocode.conversation.id'] = conversation_id
                    if duration_seconds is not None:
                        attributes['vocode.stt.duration_seconds'] = duration_seconds
                        attributes['vocode.stt.duration_ms'] = duration_seconds * 1000
                    if wpm_value is not None:
                        attributes['vocode.stt.wpm'] = wpm_value

                    # Add transcriber config if available
                    if hasattr(transcriber_instance, 'transcriber_config'):
                        cfg = transcriber_instance.transcriber_config
                        if hasattr(cfg, 'model') and getattr(cfg, 'model'):
                            attributes['vocode.stt.model'] = cfg.model
                        if hasattr(cfg, 'language') and getattr(cfg, 'language'):
                            attributes['vocode.stt.language'] = cfg.language

                    # Only create spans for final transcriptions
                    if is_final:
                        span_name = f"vocode.stt.{provider}"
                        span = tracer.start_span(span_name, attributes=attributes)
                        # Backdate span start to cover the audio duration when available
                        if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
                            try:
                                span.start_time = span.start_time - float(duration_seconds)
                            except Exception:
                                pass
                        tracer.end_span(span)

                    # Mark item to prevent double-instrumentation when also passing through produce_nonblocking
                    try:
                        setattr(item, '_ze_stt_traced', True)
                    except Exception:
                        pass
            except Exception:
                # Never fail the pipeline due to tracing
                pass

            # Forward to original queue method
            return original_put(item)

        # Bind wrapper to the specific queue instance
        setattr(queue_obj, 'put_nowait', types.MethodType(put_nowait_wrapper, queue_obj))
    
    def _wrap_conversation_terminate(self, original: Callable) -> Callable:
        """
        Wrap StreamingConversation.terminate to end the root span.
        """
        tracer = self.tracer
        
        @wraps(original)
        async def async_wrapper(conversation_instance):
            """Terminate the conversation and end the root span."""
            conversation_id = getattr(conversation_instance, 'id', 'unknown')
            
            try:
                # Call the original terminate method first
                result = await original(conversation_instance)
                
                # End the root span if it exists
                conversation_spans = _conversation_spans.get()
                if conversation_id in conversation_spans:
                    root_span = conversation_spans[conversation_id]
                    
                    # Add final metrics if available
                    if hasattr(conversation_instance, 'transcript'):
                        transcript = conversation_instance.transcript
                        if hasattr(transcript, 'event_logs'):
                            root_span.attributes["vocode.conversation.events_count"] = len(transcript.event_logs)
                    
                    logger.info(f"[Vocode] Ending conversation {conversation_id} (trace {root_span.trace_id})")
                    tracer.end_span(root_span)
                    
                    # Clean up the context
                    del conversation_spans[conversation_id]
                    _conversation_spans.set(conversation_spans)
                
                return result
                
            except Exception as exc:
                # Try to end the span even if terminate fails
                conversation_spans = _conversation_spans.get()
                if conversation_id in conversation_spans:
                    root_span = conversation_spans[conversation_id]
                    root_span.set_error(
                        code=exc.__class__.__name__,
                        message=str(exc),
                        stack=traceback.format_exc()
                    )
                    tracer.end_span(root_span)
                    del conversation_spans[conversation_id]
                    _conversation_spans.set(conversation_spans)
                raise
        
        return async_wrapper

    def _wrap_create_speech(self, original: Callable) -> Callable:
        """
        Wrap BaseSynthesizer.create_speech to trace TTS operations.
        
        This captures the complete TTS lifecycle:
        - Setup time (create_speech duration)
        - Time to first byte (TTFB) when first audio chunk arrives
        - Total streaming duration
        - All chunks and bytes transferred
        """
        tracer = self.tracer
        
        @wraps(original)
        async def async_wrapper(
            synthesizer_instance,
            message,
            chunk_size: int,
            is_first_text_chunk: bool = False,
            is_sole_text_chunk: bool = False
        ):
            """
            Wrap create_speech to track the entire TTS streaming process.
            """
            # Get the conversation ID if available
            conversation_id = None
            if hasattr(synthesizer_instance, 'streaming_conversation'):
                conversation_id = getattr(synthesizer_instance.streaming_conversation, 'id', None)
            
            # Get provider name from synthesizer class
            provider_class = synthesizer_instance.__class__.__name__
            provider = provider_class.replace('Synthesizer', '').lower()
            
            # Extract text from message
            text = message.text if hasattr(message, 'text') else str(message)
            
            # Build span attributes
            attributes = {
                "service.name": "vocode",
                "provider": provider,
                "vocode.tts.provider": provider,
                "vocode.tts.text_length": len(text),
                "vocode.tts.chunk_size": chunk_size,
                "vocode.tts.is_first_chunk": is_first_text_chunk,
                "vocode.tts.is_sole_chunk": is_sole_text_chunk,
            }
            
            if conversation_id:
                attributes["vocode.conversation.id"] = conversation_id
            
            # Add voice/model config if available
            if hasattr(synthesizer_instance, 'synthesizer_config'):
                config = synthesizer_instance.synthesizer_config
                if hasattr(config, 'voice_id'):
                    attributes["vocode.tts.voice_id"] = config.voice_id
                if hasattr(config, 'model_id'):
                    attributes["vocode.tts.model"] = config.model_id
                if hasattr(config, 'language'):
                    attributes["vocode.tts.language"] = config.language
            
            # Create span for the entire TTS operation
            span_name = f"vocode.tts.{provider}"
            span = tracer.start_span(span_name, attributes=attributes)
            
            try:
                # Track setup timing
                setup_start = time.perf_counter()
                
                # Call original method to get SynthesisResult
                result = await original(
                    synthesizer_instance,
                    message,
                    chunk_size,
                    is_first_text_chunk,
                    is_sole_text_chunk
                )
                
                # Calculate setup duration (time to create the synthesis request)
                setup_duration_ms = (time.perf_counter() - setup_start) * 1000
                span.attributes["vocode.tts.setup_ms"] = round(setup_duration_ms, 1)
                
                # Log setup completion
                logger.debug(
                    f"[Vocode TTS] {provider}: Setup completed in {setup_duration_ms:.1f}ms for {len(text)} chars"
                )
                
                # Wrap the result to track streaming metrics
                # The span will be ended when streaming completes
                wrapped_result = track_tts_streaming(span, tracer, text)(result)
                return wrapped_result
                
            except Exception as exc:
                # Record exception in span
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=traceback.format_exc()
                )
                tracer.end_span(span)
                raise
        
        return async_wrapper
    
    def _wrap_forward_from_thread(self, original: Callable) -> Callable:
        """
        Wrap BaseThreadAsyncTranscriber._forward_from_thread to trace STT operations.
        This method is used by threaded transcribers like DeepgramTranscriber.
        """
        tracer = self.tracer
        
        @wraps(original)
        async def async_wrapper(transcriber_instance):
            """
            Wrap the async _forward_from_thread method.
            This intercepts transcriptions from the thread queue.
            """
            while True:
                try:
                    # Get transcription from the thread queue
                    transcription = await transcriber_instance.output_janus_queue.async_q.get()
                    
                    # Create STT span for this transcription
                    if transcription and hasattr(transcription, 'message'):
                        # Get provider from transcriber class name
                        provider_class = transcriber_instance.__class__.__name__
                        provider = provider_class.replace('Transcriber', '').lower()
                        
                        # Get conversation ID if available
                        conversation_id = None
                        if hasattr(transcriber_instance, 'streaming_conversation'):
                            conversation_id = getattr(transcriber_instance.streaming_conversation, 'id', None)
                        
                        # Build span attributes
                        attributes = {
                            "service.name": "vocode",
                            "provider": provider,
                            "vocode.stt.provider": provider,
                            "vocode.stt.text": transcription.message,
                            "vocode.stt.text_length": len(transcription.message),
                            "vocode.stt.is_final": getattr(transcription, 'is_final', False),
                            "vocode.stt.confidence": getattr(transcription, 'confidence', 0.0),
                            "vocode.stt.is_interrupt": getattr(transcription, 'is_interrupt', False),
                        }
                        
                        if conversation_id:
                            attributes["vocode.conversation.id"] = conversation_id
                        
                        # Add timing metrics if available
                        if hasattr(transcription, 'duration_seconds'):
                            attributes["vocode.stt.duration_seconds"] = transcription.duration_seconds
                        
                        if hasattr(transcription, 'wpm') and callable(transcription.wpm):
                            try:
                                wpm_value = transcription.wpm()
                                if wpm_value is not None:
                                    attributes["vocode.stt.wpm"] = wpm_value
                            except:
                                pass
                        
                        # Create and immediately end the span
                        span_name = f"vocode.stt.{provider}"
                        span = tracer.start_span(span_name, attributes=attributes)
                        tracer.end_span(span)
                        
                        logger.debug(
                            f"[Vocode STT] {provider}: '{transcription.message[:50]}...' "
                            f"(final={getattr(transcription, 'is_final', False)}, "
                            f"interrupt={getattr(transcription, 'is_interrupt', False)})"
                        )
                    
                    # Forward to consumer as normal
                    transcriber_instance.consumer.consume_nonblocking(transcription)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in STT forwarding: {e}")
                    # Still forward even if span creation failed
                    try:
                        transcriber_instance.consumer.consume_nonblocking(transcription)
                    except:
                        pass
        
        return async_wrapper
    
    def _wrap_produce_nonblocking(self, original: Callable) -> Callable:
        logger.debug(f"[DEBUG] Wrapping produce_nonblocking: {original}")
        """
        Wrap AsyncWorker.produce_nonblocking to trace STT operations.
        
        All STT spans will be children of the conversation root span.
        """
        tracer = self.tracer
        
        @wraps(original)
        def wrapper(worker_instance, item):
            """
            Wrap the synchronous produce_nonblocking method.
            
            Args:
                worker_instance: The AsyncWorker instance (might be a transcriber)
                item: Could be a Transcription object or other data
            """
            # Check if this is a transcriber by checking the class name
            class_name = worker_instance.__class__.__name__
            is_transcriber = 'Transcriber' in class_name
            
            # Only create spans for Transcription objects from transcribers
            if not is_transcriber or not hasattr(item, 'message'):
                # Not a transcription, just pass through
                return original(worker_instance, item)
            
            logger.debug(f"[DEBUG] produce_nonblocking called: {class_name}, message='{getattr(item, 'message', 'N/A')[:30]}'...")
            # Get the conversation ID if available
            conversation_id = None
            if hasattr(worker_instance, 'streaming_conversation'):
                conversation_id = getattr(worker_instance.streaming_conversation, 'id', None)
            
            # Get provider name from transcriber class
            provider = class_name.replace('Transcriber', '').lower()
            
            # Extract rich transcription data
            message = item.message
            confidence = getattr(item, 'confidence', 0.0)
            is_final = getattr(item, 'is_final', False)
            is_interrupt = getattr(item, 'is_interrupt', False)
            bot_was_in_medias_res = getattr(item, 'bot_was_in_medias_res', False)
            duration_seconds = getattr(item, 'duration_seconds', None)
            wpm = item.wpm() if hasattr(item, 'wpm') and callable(item.wpm) and duration_seconds else None
            
            # Only create spans for meaningful transcriptions
            if not message or not message.strip():
                # Still call original for empty transcriptions (they might be important for state)
                return original(worker_instance, item)
            
            # Build span attributes with all rich data
            attributes = {
                "service.name": "vocode",
                "provider": provider,
                "vocode.stt.provider": provider,
                "vocode.stt.text": message,
                "vocode.stt.confidence": confidence,
                "vocode.stt.is_final": is_final,
                "vocode.stt.is_interrupt": is_interrupt,
                "vocode.stt.bot_was_speaking": bot_was_in_medias_res,
                "vocode.stt.message_length": len(message),
                "vocode.stt.word_count": len(message.split()),
            }
            
            if conversation_id:
                attributes["vocode.conversation.id"] = conversation_id
            
            # Add duration-based metrics if available (only on final transcriptions)
            if duration_seconds is not None:
                attributes.update({
                    "vocode.stt.duration_seconds": duration_seconds,
                    "vocode.stt.duration_ms": duration_seconds * 1000,
                })
            
            if wpm is not None:
                attributes["vocode.stt.wpm"] = round(wpm, 1)
            
            # Add transcriber config if available
            if hasattr(worker_instance, 'transcriber_config'):
                config = worker_instance.transcriber_config
                if hasattr(config, 'model'):
                    attributes["vocode.stt.model"] = config.model
                if hasattr(config, 'language'):
                    attributes["vocode.stt.language"] = config.language
            
            # Create span for transcription
            # Only create spans for final transcriptions
            if not is_final:
                return original(worker_instance, item)

            span_name = f"vocode.stt.{provider}"
            
            # The span will automatically be a child of the conversation root span
            span = tracer.start_span(span_name, attributes=attributes)
            
            # Backdate start_time by duration when possible so duration_ms is meaningful
            if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
                try:
                    span.start_time = span.start_time - float(duration_seconds)
                except Exception:
                    pass
            
            # Set input/output for the span
            input_description = f"Audio ({duration_seconds:.2f}s)" if duration_seconds else "Audio stream"
            span.set_io(
                input_data=input_description,
                output_data=message
            )
            
            try:
                # Call original method
                result = original(worker_instance, item)
                
                # Log the transcription
                logger.debug(
                    f"[Vocode STT] {provider}: '{message[:50]}{'...' if len(message) > 50 else ''}' "
                    f"(confidence={confidence:.2f}, final={is_final}, "
                    f"{'wpm=' + str(round(wpm, 0)) + ', ' if wpm else ''}"
                    f"{'interrupted' if is_interrupt else 'complete'})"
                )
                
                # End span successfully
                tracer.end_span(span)
                return result
                
            except Exception as exc:
                # Record exception in span
                span.set_error(
                    code=exc.__class__.__name__,
                    message=str(exc),
                    stack=traceback.format_exc()
                )
                tracer.end_span(span)
                raise
        
        return wrapper


# Example usage documentation
__doc__ = """
ZeroEval Integration for Vocode Voice SDK
==========================================

This integration automatically traces all Vocode voice operations including:
- Complete conversation lifecycle with all spans under one trace
- Text-to-Speech synthesis across all providers
- Speech-to-Text transcription across all providers  
- Conversation orchestration and flow
- Voice pipeline latencies

Quick Start
-----------

1. Install both packages:
   ```bash
   pip install zeroeval vocode
   ```

2. Initialize ZeroEval before using Vocode:
   ```python
   import zeroeval as ze
   
   ze.init(
       api_key="your-api-key",
       workspace_name="Voice Apps"
   )
   ```

3. Use Vocode normally - all operations are automatically traced:
   ```python
   from vocode.streaming.streaming_conversation import StreamingConversation
   from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
   from vocode.streaming.models.transcriber import DeepgramTranscriberConfig
   
   conversation = StreamingConversation(
       output_device=speaker_output,
       transcriber=DeepgramTranscriber(
           DeepgramTranscriberConfig.from_input_device(microphone_input)
       ),
       agent=ChatGPTAgent(...),
       synthesizer=ElevenLabsSynthesizer(...)
   )
   
   await conversation.start()  # Creates root span for entire conversation
   # All TTS, STT, and LLM operations are traced under the same trace
   ```

Captured Metrics
----------------

**Conversation Level:**
- conversation.id
- Agent type and model
- Transcriber and synthesizer types
- Total duration

**TTS (Text-to-Speech):**
- Provider (ElevenLabs, Azure, Google, etc.)
- Text content and length
- Voice ID and configuration
- Synthesis latency
- Cache hit/miss

**STT (Speech-to-Text):**
- Provider (Deepgram, Azure, Google, etc.)
- Transcribed text and confidence
- Final vs interim transcriptions
- Duration and WPM (words per minute)
- Interruption detection
- Model and language settings

All spans within a single conversation share the same trace ID,
making it easy to analyze the complete voice interaction flow.
"""

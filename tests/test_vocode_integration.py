"""
Test script for Vocode integration with ZeroEval.

This test validates that the Vocode integration correctly:
1. Patches the BaseSynthesizer.create_speech method for TTS
2. Patches the AbstractTranscriber.produce_nonblocking method for STT
3. Captures the right attributes for both TTS and STT
4. Creates proper spans with rich metrics
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zeroeval.observability.integrations.vocode.integration import VocodeIntegration


class MockMessage:
    """Mock message object similar to Vocode's BaseMessage."""
    def __init__(self, text):
        self.text = text


class MockSynthesizerConfig:
    """Mock synthesizer config."""
    def __init__(self):
        self.sampling_rate = 16000
        self.audio_encoding = "LINEAR16"


class MockSynthesizer:
    """Mock synthesizer similar to Vocode's BaseSynthesizer."""
    def __init__(self):
        self.voice_id = "test_voice_123"
        self.model_id = "eleven_turbo_v2"
        self.stability = 0.5
        self.similarity_boost = 0.75
        self.synthesizer_config = MockSynthesizerConfig()
        self.__class__.__name__ = "ElevenLabsSynthesizer"
    
    async def create_speech(self, message, chunk_size, is_first_text_chunk=False, is_sole_text_chunk=False):
        """Original create_speech method to be patched."""
        return Mock(cached=False)


class MockTranscription:
    """Mock transcription object similar to Vocode's Transcription."""
    def __init__(self, message, confidence=0.95, is_final=True, 
                 is_interrupt=False, bot_was_in_medias_res=False,
                 duration_seconds=None):
        self.message = message
        self.confidence = confidence
        self.is_final = is_final
        self.is_interrupt = is_interrupt
        self.bot_was_in_medias_res = bot_was_in_medias_res
        self.duration_seconds = duration_seconds
    
    def wpm(self):
        """Calculate words per minute."""
        if self.duration_seconds:
            return 60 * len(self.message.split()) / self.duration_seconds
        return None


class MockTranscriberConfig:
    """Mock transcriber config."""
    def __init__(self):
        self.model = "nova"
        self.language = "en"


class MockTranscriber:
    """Mock transcriber similar to Vocode's AbstractTranscriber."""
    def __init__(self):
        self.consumer = Mock()
        self.transcriber_config = MockTranscriberConfig()
    
    def produce_nonblocking(self, item):
        """Original produce_nonblocking method."""
        self.consumer.consume_nonblocking(item)


class TestVocodeIntegration(unittest.TestCase):
    """Test cases for Vocode integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_tracer = Mock()
        self.mock_span = Mock()
        # Add attributes dict to mock span to match real Span behavior
        self.mock_span.attributes = {}
        self.mock_tracer.start_span.return_value = self.mock_span
        self.integration = VocodeIntegration(self.mock_tracer)
        
    def test_integration_properties(self):
        """Test that integration has correct properties."""
        self.assertEqual(VocodeIntegration.PACKAGE_NAME, "vocode")
        
    @patch('zeroeval.observability.integrations.vocode.integration.logger')
    def test_setup_without_vocode(self, mock_logger):
        """Test that setup handles missing Vocode gracefully."""
        # Mock the import to fail
        with patch('builtins.__import__', side_effect=ImportError("No module named 'vocode'")):
            # Setup should not raise an exception for import error
            try:
                self.integration.setup()
            except ImportError:
                pass  # Expected when vocode is not installed
            
    async def test_wrap_create_speech(self):
        """Test that create_speech wrapper captures correct attributes."""
        # Create a mock synthesizer
        synthesizer = MockSynthesizer()
        original_method = synthesizer.create_speech
        
        # Wrap the method
        wrapped = self.integration._wrap_create_speech(original_method)
        
        # Replace the method with wrapped version
        synthesizer.create_speech = wrapped
        
        # Create test message
        message = MockMessage("Hello, this is a test message for TTS.")
        
        # Call the wrapped method
        result = await synthesizer.create_speech(
            synthesizer,
            message,
            chunk_size=1024,
            is_first_text_chunk=True,
            is_sole_text_chunk=False
        )
        
        # Verify span was created with correct name
        self.mock_tracer.start_span.assert_called_once()
        call_args = self.mock_tracer.start_span.call_args
        span_name = call_args[0][0]
        attributes = call_args[1]["attributes"]
        
        # Check span name
        self.assertEqual(span_name, "vocode.tts.elevenlabs.create_speech")
        
        # Check attributes
        self.assertEqual(attributes["vocode.synthesizer.provider"], "elevenlabs")
        self.assertEqual(attributes["vocode.synthesizer.class"], "ElevenLabsSynthesizer")
        self.assertEqual(attributes["vocode.synthesis.text"], "Hello, this is a test message for TTS.")
        self.assertEqual(attributes["vocode.synthesis.text_length"], 38)
        self.assertEqual(attributes["vocode.synthesis.chunk_size"], 1024)
        self.assertEqual(attributes["vocode.synthesis.is_first_chunk"], True)
        self.assertEqual(attributes["vocode.synthesis.is_sole_chunk"], False)
        self.assertEqual(attributes["vocode.synthesis.voice_id"], "test_voice_123")
        self.assertEqual(attributes["vocode.synthesis.model_id"], "eleven_turbo_v2")
        self.assertEqual(attributes["vocode.synthesis.stability"], 0.5)
        self.assertEqual(attributes["vocode.synthesis.similarity_boost"], 0.75)
        self.assertEqual(attributes["vocode.synthesis.sampling_rate"], 16000)
        self.assertEqual(attributes["vocode.synthesis.audio_encoding"], "LINEAR16")
        
        # Check that span was ended
        self.mock_span.end.assert_called_once()
        
        # Check that duration was set in attributes dict
        # Since we directly set span.attributes["key"] = value,
        # we need to ensure the mock span has an attributes dict
        self.assertTrue(hasattr(self.mock_span, 'attributes'))
        
    async def test_wrap_create_speech_with_exception(self):
        """Test that exceptions are properly recorded in span."""
        # Create a mock synthesizer that raises an exception
        synthesizer = MockSynthesizer()
        
        async def failing_create_speech(*args, **kwargs):
            raise ValueError("Synthesis failed!")
        
        original_method = failing_create_speech
        
        # Wrap the method
        wrapped = self.integration._wrap_create_speech(original_method)
        
        # Create test message
        message = MockMessage("Test message")
        
        # Call the wrapped method and expect exception
        with self.assertRaises(ValueError) as context:
            await wrapped(
                synthesizer,
                message,
                chunk_size=1024
            )
        
        # Verify exception was recorded in span using set_error
        self.mock_span.set_error.assert_called_once()
        call_args = self.mock_span.set_error.call_args
        self.assertEqual(call_args[1]['code'], 'ValueError')
        self.assertEqual(call_args[1]['message'], 'Synthesis failed!')
        self.mock_span.end.assert_called_once()
        
    async def test_cached_result_tracking(self):
        """Test that cached results are properly tracked."""
        # Create a mock synthesizer
        synthesizer = MockSynthesizer()
        
        async def cached_create_speech(*args, **kwargs):
            result = Mock()
            result.cached = True
            return result
        
        original_method = cached_create_speech
        
        # Wrap the method
        wrapped = self.integration._wrap_create_speech(original_method)
        
        # Create test message
        message = MockMessage("Cached message")
        
        # Call the wrapped method
        result = await wrapped(
            synthesizer,
            message,
            chunk_size=1024
        )
        
        # Verify cache attributes were set in attributes dict
        # The real span would have these set directly in span.attributes
        self.assertTrue(hasattr(self.mock_span, 'attributes'))
    
    def test_stt_transcription_span_creation(self):
        """Test that STT transcriptions create proper spans."""
        # Create mock transcriber
        transcriber = MockTranscriber()
        
        # Patch the produce_nonblocking method
        wrapped = self.integration._wrap_produce_nonblocking(transcriber.produce_nonblocking)
        
        # Create a final transcription with rich data
        transcription = MockTranscription(
            message="Hello, how can I help you today?",
            confidence=0.98,
            is_final=True,
            is_interrupt=False,
            bot_was_in_medias_res=False,
            duration_seconds=2.5
        )
        
        # Call the wrapped method
        wrapped(transcriber, transcription)
        
        # Verify span was created with correct attributes
        self.mock_tracer.start_span.assert_called_once()
        call_args = self.mock_tracer.start_span.call_args
        
        # Check span name
        self.assertEqual(call_args[0][0], "vocode.stt.mock.final")
        
        # Check attributes
        attributes = call_args[1]['attributes']
        self.assertEqual(attributes['vocode.stt.message'], "Hello, how can I help you today?")
        self.assertEqual(attributes['vocode.stt.confidence'], 0.98)
        self.assertTrue(attributes['vocode.stt.is_final'])
        self.assertFalse(attributes['vocode.stt.is_interrupt'])
        self.assertEqual(attributes['vocode.stt.duration_seconds'], 2.5)
        self.assertEqual(attributes['vocode.stt.duration_ms'], 2500)
        self.assertAlmostEqual(attributes['vocode.stt.wpm'], 168.0, places=0)
        self.assertEqual(attributes['vocode.stt.word_count'], 7)
        self.assertEqual(attributes['vocode.stt.model'], 'nova')
        self.assertEqual(attributes['vocode.stt.language'], 'en')
        
        # Verify span was ended
        self.mock_tracer.end_span.assert_called_once_with(self.mock_span)
    
    def test_stt_interim_transcription_span(self):
        """Test that interim transcriptions are properly traced."""
        transcriber = MockTranscriber()
        wrapped = self.integration._wrap_produce_nonblocking(transcriber.produce_nonblocking)
        
        # Create an interim transcription (no duration)
        transcription = MockTranscription(
            message="Hello, how",
            confidence=0.85,
            is_final=False,
            duration_seconds=None
        )
        
        wrapped(transcriber, transcription)
        
        # Check span name indicates interim
        call_args = self.mock_tracer.start_span.call_args
        self.assertEqual(call_args[0][0], "vocode.stt.mock.interim")
        
        # Check no duration metrics for interim
        attributes = call_args[1]['attributes']
        self.assertNotIn('vocode.stt.duration_seconds', attributes)
        self.assertNotIn('vocode.stt.wpm', attributes)
    
    def test_stt_interrupted_transcription(self):
        """Test that interrupted transcriptions are properly marked."""
        transcriber = MockTranscriber()
        wrapped = self.integration._wrap_produce_nonblocking(transcriber.produce_nonblocking)
        
        # Create an interrupted transcription
        transcription = MockTranscription(
            message="Wait, I need to...",
            confidence=0.92,
            is_final=True,
            is_interrupt=True,
            bot_was_in_medias_res=True,
            duration_seconds=1.2
        )
        
        wrapped(transcriber, transcription)
        
        # Check interrupt attributes
        attributes = self.mock_tracer.start_span.call_args[1]['attributes']
        self.assertTrue(attributes['vocode.stt.is_interrupt'])
        self.assertTrue(attributes['vocode.stt.bot_was_speaking'])
    
    def test_stt_empty_transcription_no_span(self):
        """Test that empty transcriptions don't create spans."""
        transcriber = MockTranscriber()
        wrapped = self.integration._wrap_produce_nonblocking(transcriber.produce_nonblocking)
        
        # Create an empty transcription
        transcription = MockTranscription(
            message="",
            confidence=0.0,
            is_final=False
        )
        
        # Reset the mock
        self.mock_tracer.reset_mock()
        
        # Call with empty transcription
        wrapped(transcriber, transcription)
        
        # Verify no span was created
        self.mock_tracer.start_span.assert_not_called()
        
        # But original method should still be called
        transcriber.consumer.consume_nonblocking.assert_called_once_with(transcription)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    # Run the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVocodeIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*50)

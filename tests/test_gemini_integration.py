"""Tests for the Gemini integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from zeroeval.observability.integrations.gemini.integration import GeminiIntegration


class TestGeminiIntegration:
    """Test suite for GeminiIntegration."""
    
    def test_is_available_when_package_exists(self):
        """Test that is_available returns True when google.genai is installed."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()
            assert GeminiIntegration.is_available() is True
            mock_import.assert_called_once_with('google.genai')
    
    def test_is_available_when_package_missing(self):
        """Test that is_available returns False when google.genai is not installed."""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError()
            assert GeminiIntegration.is_available() is False
    
    def test_setup_patches_client_init(self):
        """Test that setup correctly patches the Client.__init__ method."""
        # Mock the google.genai module
        mock_genai = Mock()
        mock_client_class = Mock()
        mock_genai.Client = mock_client_class
        
        with patch.dict('sys.modules', {'google': Mock(genai=mock_genai), 'google.genai': mock_genai}):
            # Create integration with a mock tracer
            mock_tracer = Mock()
            integration = GeminiIntegration(mock_tracer)
            
            # Run setup
            integration.setup()
            
            # Verify that Client.__init__ was patched
            assert mock_client_class.__init__ in integration._original_functions.values()
    
    def test_serialize_contents_handles_strings(self):
        """Test that _serialize_contents handles string inputs correctly."""
        integration = GeminiIntegration(Mock())
        
        result = integration._serialize_contents("Hello, world!")
        assert result == "Hello, world!"
    
    def test_serialize_contents_handles_lists(self):
        """Test that _serialize_contents handles list inputs correctly."""
        integration = GeminiIntegration(Mock())
        
        # Test with list of strings
        result = integration._serialize_contents(["Hello", "World"])
        assert result == ["Hello", "World"]
        
        # Test with mixed list
        mock_obj = Mock()
        mock_obj.__dict__ = {"text": "test", "role": "user"}
        result = integration._serialize_contents(["Hello", mock_obj])
        assert len(result) == 2
        assert result[0] == "Hello"
        assert isinstance(result[1], dict) or isinstance(result[1], str)
    
    def test_extract_config_attributes(self):
        """Test that _extract_config_attributes correctly extracts configuration."""
        integration = GeminiIntegration(Mock())
        
        # Mock config object
        mock_config = Mock()
        mock_config.temperature = 0.7
        mock_config.max_output_tokens = 100
        mock_config.top_p = 0.9
        mock_config.response_mime_type = "application/json"
        
        result = integration._extract_config_attributes(mock_config)
        
        assert result['temperature'] == 0.7
        assert result['max_output_tokens'] == 100
        assert result['top_p'] == 0.9
        assert result['response_mime_type'] == "application/json"
    
    def test_extract_config_attributes_with_tools(self):
        """Test that _extract_config_attributes correctly extracts tool information."""
        integration = GeminiIntegration(Mock())
        
        # Mock function declaration
        mock_func_decl = Mock()
        mock_func_decl.name = "get_weather"
        mock_func_decl.description = "Get weather information"
        
        # Mock tool with function declarations
        mock_tool = Mock()
        mock_tool.function_declarations = [mock_func_decl]
        
        # Mock config with tools
        mock_config = Mock()
        mock_config.tools = [mock_tool]
        
        # Also test with a callable
        def test_function():
            """Test function docstring"""
            pass
        
        mock_config.tools.append(test_function)
        
        result = integration._extract_config_attributes(mock_config)
        
        assert 'tools' in result
        assert len(result['tools']) == 2
        assert result['tools'][0]['name'] == 'get_weather'
        assert result['tools'][0]['description'] == 'Get weather information'
        assert result['tools'][1]['name'] == 'test_function'
        assert result['tools'][1]['description'] == 'Test function docstring'
    
    def test_object_to_dict_with_to_dict_method(self):
        """Test _object_to_dict when object has to_dict method."""
        integration = GeminiIntegration(Mock())
        
        mock_obj = Mock()
        mock_obj.to_dict.return_value = {"key": "value"}
        
        result = integration._object_to_dict(mock_obj)
        assert result == {"key": "value"}
        mock_obj.to_dict.assert_called_once()
    
    def test_object_to_dict_with_dict_attribute(self):
        """Test _object_to_dict when object has __dict__ attribute."""
        integration = GeminiIntegration(Mock())
        
        class TestObj:
            def __init__(self):
                self.public_attr = "public"
                self._private_attr = "private"
                self.nested = Mock()
                self.nested.__dict__ = {"inner": "value"}
        
        obj = TestObj()
        result = integration._object_to_dict(obj)
        
        assert "public_attr" in result
        assert result["public_attr"] == "public"
        assert "_private_attr" not in result  # Private attrs excluded
        assert "nested" in result
        assert isinstance(result["nested"], dict)
    
    @patch('time.time')
    def test_wrap_generate_content_success(self, mock_time):
        """Test successful wrapping of generate_content method."""
        mock_time.side_effect = [1000, 1001]  # Start and end time
        
        # Setup mocks
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        
        integration = GeminiIntegration(mock_tracer)
        
        # Mock response
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_part = Mock()
        mock_part.text = "The sky is blue."
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_response.candidates = [mock_candidate]
        
        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.total_token_count = 30
        mock_response.usage_metadata = mock_usage
        
        # Mock original function
        original_func = Mock(return_value=mock_response)
        
        # Wrap the function
        wrapped_func = integration._wrap_generate_content(original_func)
        
        # Call wrapped function
        result = wrapped_func(
            model="gemini-2.0-flash-001",
            contents="Why is the sky blue?",
            config=None
        )
        
        # Verify span was created with correct attributes
        mock_tracer.start_span.assert_called_once()
        call_args = mock_tracer.start_span.call_args
        assert call_args[1]["name"] == "gemini.models.generate_content"
        assert call_args[1]["kind"] == "llm"
        assert call_args[1]["attributes"]["model"] == "gemini-2.0-flash-001"
        assert call_args[1]["attributes"]["streaming"] is False
        
        # Verify span attributes were set
        assert mock_span.attributes["inputTokens"] == 10
        assert mock_span.attributes["outputTokens"] == 20
        assert mock_span.attributes["totalTokens"] == 30
        assert mock_span.attributes["finish_reason"] == "STOP"
        assert "throughput" in mock_span.attributes
        
        # Verify span IO was set
        mock_span.set_io.assert_called_once()
        
        # Verify span was ended
        mock_tracer.end_span.assert_called_once_with(mock_span)
        
        # Verify result is returned unchanged
        assert result == mock_response
    
    def test_wrap_generate_content_error_handling(self):
        """Test error handling in wrapped generate_content method."""
        # Setup mocks
        mock_tracer = Mock()
        mock_span = Mock()
        mock_tracer.start_span.return_value = mock_span
        
        integration = GeminiIntegration(mock_tracer)
        
        # Mock original function that raises an error
        original_func = Mock(side_effect=ValueError("Test error"))
        
        # Wrap the function
        wrapped_func = integration._wrap_generate_content(original_func)
        
        # Call wrapped function and expect error
        with pytest.raises(ValueError, match="Test error"):
            wrapped_func(
                model="gemini-2.0-flash-001",
                contents="Test content"
            )
        
        # Verify error was recorded in span
        mock_span.set_error.assert_called_once()
        error_call = mock_span.set_error.call_args
        assert error_call[1]["code"] == "ValueError"
        assert error_call[1]["message"] == "Test error"
        
        # Verify span was ended even with error
        mock_tracer.end_span.assert_called_once_with(mock_span)
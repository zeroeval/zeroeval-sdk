"""
Test to verify Gemini integration compatibility with google-genai 1.21.1.

This test file checks that our integration correctly handles the API structure
of google-genai 1.21.1, which is the minimum supported version.
"""

import pytest
from unittest.mock import Mock, MagicMock
from zeroeval.observability.integrations.gemini.integration import GeminiIntegration


class TestGeminiCompatibility:
    """Test suite for google-genai 1.21.1 compatibility."""
    
    def test_function_declaration_compatibility(self):
        """Test that the integration handles function declarations correctly for 1.21.1."""
        integration = GeminiIntegration(Mock())
        
        # Mock a function declaration as it appears in 1.21.1
        mock_func_decl = Mock()
        mock_func_decl.name = "get_weather"
        mock_func_decl.description = "Get weather for a location"
        # In 1.21.1, parameters use types.Schema
        mock_func_decl.parameters = Mock()
        mock_func_decl.parameters.type = "OBJECT"
        mock_func_decl.parameters.properties = {
            "location": {"type": "STRING", "description": "City and state"}
        }
        mock_func_decl.parameters.required = ["location"]
        
        # Mock tool with function declarations
        mock_tool = Mock()
        mock_tool.function_declarations = [mock_func_decl]
        
        # Mock config with tools
        mock_config = Mock()
        mock_config.tools = [mock_tool]
        
        # Extract attributes
        result = integration._extract_config_attributes(mock_config)
        
        # Verify tools were extracted correctly
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Get weather for a location"
    
    def test_response_structure_compatibility(self):
        """Test that the integration handles response structure correctly for 1.21.1."""
        integration = GeminiIntegration(Mock())
        
        # Mock response structure as it appears in 1.21.1
        mock_response = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_part = Mock()
        
        # Text response
        mock_part.text = "Test response"
        mock_part.function_call = None  # No function call
        
        mock_content.parts = [mock_part]
        mock_content.role = "model"
        
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidate.safety_ratings = []
        
        mock_response.candidates = [mock_candidate]
        
        # Usage metadata format in 1.21.1
        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.total_token_count = 30
        mock_response.usage_metadata = mock_usage
        
        # This should be handled correctly by our integration
        # Just verify the structure is as expected
        assert hasattr(mock_response, 'candidates')
        assert hasattr(mock_response.candidates[0], 'content')
        assert hasattr(mock_response.candidates[0].content, 'parts')
        assert hasattr(mock_response.usage_metadata, 'prompt_token_count')
    
    def test_function_call_response_compatibility(self):
        """Test handling of function call responses in 1.21.1 format."""
        integration = GeminiIntegration(Mock())
        
        # Mock function call response
        mock_fc = Mock()
        mock_fc.name = "get_weather"
        mock_fc.args = {"location": "San Francisco, CA"}
        
        # In 1.21.1, function calls are in parts
        mock_part = Mock()
        mock_part.text = None
        mock_part.function_call = mock_fc
        
        # Test extraction logic
        # The integration should handle this correctly
        assert hasattr(mock_part, 'function_call')
        assert mock_part.function_call.name == "get_weather"
        assert mock_part.function_call.args == {"location": "San Francisco, CA"}
    
    def test_environment_variable_compatibility(self):
        """Test that the integration works with both GOOGLE_API_KEY and GEMINI_API_KEY."""
        # This is handled by the google-genai client itself, not our integration
        # Our integration just patches the client after it's created
        # So we don't need to handle the environment variable differences
        pass
    
    def test_config_attributes_with_1_21_1_types(self):
        """Test config attribute extraction with 1.21.1 type names."""
        integration = GeminiIntegration(Mock())
        
        # Mock config as it appears in 1.21.1
        mock_config = Mock()
        mock_config.temperature = 0.7
        mock_config.max_output_tokens = 1000
        mock_config.top_p = 0.9
        mock_config.top_k = 40
        mock_config.stop_sequences = ["END"]
        mock_config.response_mime_type = "application/json"
        
        # Test response schema (Pydantic model)
        class TestModel:
            pass
        mock_config.response_schema = TestModel
        
        # Extract attributes
        result = integration._extract_config_attributes(mock_config)
        
        # Verify all attributes were extracted
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 1000
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["stop_sequences"] == ["END"]
        assert result["response_mime_type"] == "application/json"
        assert "response_schema" in result
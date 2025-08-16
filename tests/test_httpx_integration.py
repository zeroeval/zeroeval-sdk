"""
Test the httpx integration with mock Gemini API responses.
"""

import json
import pytest
import httpx
from unittest.mock import patch, MagicMock
from zeroeval import ze
from zeroeval.observability.integrations.httpx.integration import HttpxIntegration


def test_httpx_integration_setup():
    """Test that httpx integration can be set up."""
    # Create a mock tracer
    mock_tracer = MagicMock()
    
    # Create and setup the integration
    integration = HttpxIntegration(mock_tracer)
    integration.setup()
    
    # Verify that httpx methods are patched
    assert hasattr(httpx.Client.request, "__ze_patched__")
    assert hasattr(httpx.AsyncClient.request, "__ze_patched__")


def test_gemini_url_pattern_matching():
    """Test that the Gemini URL pattern correctly identifies API endpoints."""
    mock_tracer = MagicMock()
    integration = HttpxIntegration(mock_tracer)
    
    # Valid Gemini URLs
    valid_urls = [
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
    ]
    
    for url in valid_urls:
        assert integration._should_trace_request(url), f"Should trace: {url}"
    
    # Invalid URLs that should not be traced
    invalid_urls = [
        "https://api.openai.com/v1/chat/completions",
        "https://example.com/api",
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://google.com",
    ]
    
    for url in invalid_urls:
        assert not integration._should_trace_request(url), f"Should not trace: {url}"


def test_model_extraction_from_url():
    """Test extracting model name from Gemini URL."""
    mock_tracer = MagicMock()
    integration = HttpxIntegration(mock_tracer)
    
    test_cases = [
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", "gemini-1.5-flash"),
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent", "gemini-pro"),
        ("https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro-latest:generateContent", "gemini-1.5-pro-latest"),
    ]
    
    for url, expected_model in test_cases:
        assert integration._extract_model_from_url(url) == expected_model


def test_operation_extraction_from_url():
    """Test extracting operation name from Gemini URL."""
    mock_tracer = MagicMock()
    integration = HttpxIntegration(mock_tracer)
    
    test_cases = [
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent", "generateContent"),
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent", "streamGenerateContent"),
    ]
    
    for url, expected_operation in test_cases:
        assert integration._extract_operation_from_url(url) == expected_operation


def test_gemini_request_parsing():
    """Test parsing of Gemini API request payloads."""
    mock_tracer = MagicMock()
    integration = HttpxIntegration(mock_tracer)
    
    # Test basic request
    request_data = {
        "contents": [{"parts": [{"text": "Hello"}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100,
            "topP": 0.9,
            "topK": 40
        }
    }
    
    attrs = integration._parse_gemini_request(request_data)
    assert attrs["contents"] == request_data["contents"]
    assert attrs["temperature"] == 0.7
    assert attrs["max_output_tokens"] == 100
    assert attrs["top_p"] == 0.9
    assert attrs["top_k"] == 40
    
    # Test request with tools
    request_with_tools = {
        "contents": [{"parts": [{"text": "Hello"}]}],
        "tools": [{
            "functionDeclarations": [{
                "name": "get_weather",
                "description": "Get weather information"
            }]
        }],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "AUTO"
            }
        }
    }
    
    attrs = integration._parse_gemini_request(request_with_tools)
    assert "tools" in attrs
    assert len(attrs["tools"]) == 1
    assert attrs["tools"][0]["name"] == "get_weather"
    assert attrs["tool_calling_mode"] == "AUTO"


def test_gemini_response_parsing():
    """Test parsing of Gemini API response payloads."""
    mock_tracer = MagicMock()
    integration = HttpxIntegration(mock_tracer)
    
    # Test basic response
    response_data = {
        "candidates": [{
            "content": {
                "parts": [{"text": "Hello, world!"}]
            },
            "finishReason": "STOP",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"}
            ]
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30
        },
        "modelVersion": "gemini-1.5-flash-001",
        "responseId": "abc123"
    }
    
    attrs, output = integration._parse_gemini_response(response_data)
    assert output == "Hello, world!"
    assert attrs["finish_reason"] == "STOP"
    assert attrs["inputTokens"] == 10
    assert attrs["outputTokens"] == 20
    assert attrs["totalTokens"] == 30
    assert attrs["model_version"] == "gemini-1.5-flash-001"
    assert attrs["response_id"] == "abc123"
    assert len(attrs["safety_ratings"]) == 1
    
    # Test response with function call
    response_with_function = {
        "candidates": [{
            "content": {
                "parts": [{
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "San Francisco"}
                    }
                }]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 5,
            "totalTokenCount": 20
        }
    }
    
    attrs, output = integration._parse_gemini_response(response_with_function)
    assert "function_calls" in attrs
    assert len(attrs["function_calls"]) == 1
    assert attrs["function_calls"][0]["name"] == "get_weather"
    assert attrs["function_calls"][0]["args"]["location"] == "San Francisco"
    # When there's only a function call, output should be the JSON representation
    assert "get_weather" in output


@pytest.mark.asyncio
async def test_async_request_tracing():
    """Test that async requests are properly traced."""
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_span.return_value = mock_span
    
    integration = HttpxIntegration(mock_tracer)
    integration.setup()
    
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps({
        "candidates": [{
            "content": {"parts": [{"text": "Test response"}]}
        }],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 3,
            "totalTokenCount": 8
        }
    })
    
    # Patch the original httpx request
    with patch("httpx.AsyncClient.request", new=MagicMock(return_value=mock_response)):
        # Re-apply our wrapper
        original_method = httpx.AsyncClient.request
        wrapped_method = integration._wrap_request_async(original_method)
        
        # Make a request that should be traced
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        response = await wrapped_method(
            None,  # self (client instance)
            "POST",
            url,
            json={"contents": [{"parts": [{"text": "Hello"}]}]}
        )
    
    # Verify span was created
    mock_tracer.start_span.assert_called_once()
    call_args = mock_tracer.start_span.call_args
    assert call_args[0][0] == "gemini.models.generateContent"
    assert call_args[1]["kind"] == "llm"
    assert call_args[1]["attributes"]["model"] == "gemini-1.5-flash"
    
    # Verify span was ended
    mock_tracer.end_span.assert_called_once_with(mock_span)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

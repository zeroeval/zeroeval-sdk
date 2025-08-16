# HttpX Integration

This integration provides network-level tracing for HTTP requests made through the `httpx` library, with special support for LLM provider APIs.

## Features

- **Automatic Request Interception**: Patches httpx Client and AsyncClient to intercept all HTTP requests
- **Selective Tracing**: Only traces requests to supported LLM provider endpoints
- **Gemini API Support**: Full support for Google Gemini REST API including:
  - generateContent endpoint
  - streamGenerateContent endpoint (with SSE streaming support)
  - Function/tool calling detection
  - Usage metadata extraction
  - Safety ratings capture

## How It Works

The integration works by:

1. Patching httpx's `request` method on both sync and async clients
2. Filtering requests by URL pattern to identify LLM API calls
3. Parsing request/response payloads to extract relevant information
4. Creating LLM spans with appropriate attributes
5. Creating child tool spans for function calls

## Supported Endpoints

Currently supports:

- **Google Gemini API**: `https://generativelanguage.googleapis.com/v*/models/*:generateContent`
- **Google Gemini Streaming**: `https://generativelanguage.googleapis.com/v*/models/*:streamGenerateContent`

## Usage

The integration is automatically enabled when httpx is installed. You can also explicitly enable it:

```python
import ze
ze.init(
    api_key="your_api_key",
    integrations=["HttpxIntegration"]
)
```

Or disable it:

```python
ze.init(
    api_key="your_api_key",
    disabled_integrations=["httpx"]
)
```

## Example

```python
import httpx
import ze

ze.init(api_key="your_key")

# Make a direct API call to Gemini
with httpx.Client() as client:
    response = client.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        headers={"x-goog-api-key": "your_gemini_key"},
        json={
            "contents": [{"parts": [{"text": "Hello"}]}],
            "generationConfig": {"temperature": 0.7}
        }
    )
```

## Captured Attributes

For Gemini API calls, the integration captures:

**Request Attributes:**

- Model name
- Contents (messages)
- Generation config (temperature, max_output_tokens, etc.)
- Tools and tool configuration
- System instruction
- Cached content reference

**Response Attributes:**

- Output text or function calls
- Token usage (input, output, total)
- Finish reason
- Safety ratings
- Model version
- Response ID
- Throughput (chars/second)

## Streaming Support

The integration fully supports streaming responses through Server-Sent Events (SSE). It accumulates chunks and creates the span once streaming completes, including total token usage.

## Extensibility

The integration is designed to be easily extensible to support other LLM providers. To add support for a new provider:

1. Add URL pattern matching in `_should_trace_request()`
2. Add request parsing in a new method like `_parse_provider_request()`
3. Add response parsing in a new method like `_parse_provider_response()`
4. Update the wrapper methods to route to appropriate parsers based on URL

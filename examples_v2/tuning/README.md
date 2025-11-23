# Prompt Tuning Examples

This directory contains examples demonstrating ZeroEval's prompt tuning and optimization features.

## Core Concepts

Prompt tuning in ZeroEval works through a feedback loop:

1. **Define Prompt**: Use `ze.prompt()` to register a prompt and bind variables.
2. **Trace Execution**: Run your agent; the SDK automatically traces the inputs and outputs.
3. **Send Feedback**: Use `ze.send_feedback()` (or the direct API) to signal what was good or bad about the completion.
4. **Optimize**: ZeroEval (and integrated optimizers like DSPy) uses this feedback to generate better prompt versions.

## Examples

### 1. Customer Support Agent (`customer_support_agent.py`)

A simple example of a support agent that uses `ze.prompt()` for versioned, managed prompts. This demonstrates the basic setup without the automated feedback loop.

### 2. Customer Support Agent with SDK Feedback (`bookstore_agent_with_feedback.py`)

An advanced example that implements a complete automated feedback loop using the ZeroEval SDK.

**Key Features:**

- **Automated Evaluator**: Uses a powerful model (GPT-4o) to grade the agent's responses.
- **Feedback Submission**: Uses `ze.send_feedback()` to programmatically submit the evaluator's scores (thumbs up/down) and reasoning.
- **Metadata Tracking**: Attaches metadata (like scores and evaluator model) to the feedback.

**Run it:**

```bash
python tuning/bookstore_agent_with_feedback.py
```

### 3. Customer Support Agent with API Feedback (`bookstore_agent_with_api_feedback.py`)

Demonstrates how to submit feedback using direct HTTP calls to the ZeroEval API, bypassing the SDK's `ze.send_feedback` helper. This is useful for frontend applications or systems where the SDK cannot be installed.

**Key Features:**

- **Direct API Integration**: Uses `requests` to hit the `/v1/prompts/{slug}/completions/{id}/feedback` endpoint.
- **Payload Structure**: Shows exactly what JSON payload the backend expects.
- **Flexible Integration**: Ideal for custom pipelines or non-Python environments.

**Run it:**

```bash
python tuning/bookstore_agent_with_api_feedback.py
```

## Setup

Ensure you have your `.env` file set up in the parent directory with:

- `ZEROEVAL_API_KEY`: Your ZeroEval API key (required, starts with `sk_ze_...`)
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ZEROEVAL_API_URL`: (Optional) URL of your ZeroEval instance (default: `http://localhost:8000`)

**Important**: All examples now pull credentials from environment variables. Never commit hardcoded API keys to version control.

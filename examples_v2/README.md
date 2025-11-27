# ZeroEval Examples v2

This directory contains organized, focused examples for ZeroEval SDK features.

## Directory Structure

- **`tracing/`** - Examples for observability and tracing
  - Basic OpenAI tracing
  - Custom span creation
  - Advanced tracing patterns

- **`ab_testing/`** - Examples for A/B testing with ze.choose()
  - Model comparison testing
  - Weighted variant selection
  - Automatic choice tracking

- **`tuning/`** - Examples for Prompt Tuning and Optimization
  - Customer support agent with feedback loop
  - Prompt versioning with ze.prompt()

- **`structured_outputs/`** - Examples for PydanticAI agents
  - Customer support agent with typed outputs using `agent.iter()`
  - Tool definitions and streaming execution chunks
  - Structured action suggestions and sentiment detection

## Getting Started

1. **Install dependencies**:

   ```bash
   # Using pip
   pip install zeroeval openai python-dotenv

   # Or using uv (recommended for py-sdk development)
   uv add zeroeval openai python-dotenv
   ```

2. **Set up your environment**:

   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. **Explore the examples**:
   ```bash
   # Tracing examples
   cd tracing/
   uv run python openai_basic.py

   # A/B testing examples
   cd ab_testing/
   uv run python openai_ab_test.py
   ```

Each subdirectory contains its own README with specific setup instructions and explanations.

## Philosophy

These examples follow these principles:

- **Simple and focused**: Each example demonstrates one clear concept
- **Well-documented**: Extensive comments and explanations
- **Production-ready**: Code patterns you can use in real applications
- **Organized**: Grouped by feature area for easy discovery

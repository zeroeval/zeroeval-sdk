# Structured Outputs Examples

Examples demonstrating how to use ZeroEval with PydanticAI for structured agent outputs.

## Overview

This example combines PydanticAI with ZeroEval's prompt management:

- **`ze.prompt()`**: Manages prompts with versioning, variable interpolation, and optimization
- **Typed outputs**: Pydantic models ensure valid, structured responses
- **Tool support**: Define tools the agent can use
- **Streaming iteration**: Process agent execution step-by-step with `agent.iter()`
- **ZeroEval tracing**: Automatic observability for all agent interactions

## Examples

### `bookstore_agent_pydantic.py`

A customer support agent built with PydanticAI that:
- Returns structured responses with sentiment, confidence, and actions
- Uses tools to check orders, recommend books, and apply discounts
- Streams execution chunks for real-time processing
- Maintains conversation history across turns

```bash
# Install dependencies
pip install pydantic-ai zeroeval python-dotenv

# Run the example
cd examples_v2
python structured_outputs/bookstore_agent_pydantic.py
```

## Key Concepts

### 1. Define Output Models

```python
from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    message: str = Field(description="Response to show the customer")
    sentiment: str = Field(description="Customer sentiment")
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human_review: bool
```

### 2. Use `ze.prompt()` for Prompt Management

```python
import zeroeval as ze

# Define prompt template with {{variable}} syntax
PROMPT_TEMPLATE = """You are a helpful assistant.
Customer: {{user_name}}
Membership: {{membership}}
"""

# Register and interpolate the prompt
system_prompt = ze.prompt(
    name="my-agent-prompt",
    content=PROMPT_TEMPLATE,
    variables={
        "user_name": user_context.name,
        "membership": user_context.membership
    }
)
```

### 3. Create the Agent

```python
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    output_type=AgentOutput,
    system_prompt=system_prompt,  # Use ze.prompt() output
    deps_type=UserContext,
)

# Define tools
@agent.tool
async def check_order(ctx, order_id: str) -> str:
    return f"Order {order_id} is on its way!"
```

### 4. Iterate Through Agent Execution

```python
async with agent.iter(user_input, deps=user_context, message_history=history) as agent_run:
    async for node in agent_run:
        if Agent.is_user_prompt_node(node):
            # Handle user prompt
            pass
        elif Agent.is_call_tools_node(node):
            # Handle tool calls
            for part in node.model_response.parts:
                if isinstance(part, ToolCallPart):
                    print(f"Calling tool: {part.tool_name}")
        elif Agent.is_model_request_node(node):
            # Handle tool results
            pass
        elif Agent.is_end_node(node):
            # Get final output
            output: AgentOutput = node.data.output
            print(output.message)
```

### 5. ZeroEval Tracing

ZeroEval automatically captures:
- Agent initialization and configuration
- Tool calls and results
- Model requests and responses
- Structured output data

## Node Types

| Node Type | Description |
|-----------|-------------|
| `is_user_prompt_node` | User's input message |
| `is_call_tools_node` | Agent requesting tool execution |
| `is_model_request_node` | Tool results returned to model |
| `is_end_node` | Final structured output |

## Requirements

```
pydantic>=2.0
pydantic-ai>=0.1.0
zeroeval
python-dotenv
```

Set these environment variables:
- `ZEROEVAL_API_KEY`
- `OPENAI_API_KEY`

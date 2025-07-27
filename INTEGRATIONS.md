# ZeroEval Integrations

ZeroEval automatically instruments popular AI/ML frameworks to provide comprehensive observability without manual instrumentation. When you initialize the ZeroEval tracer, it automatically detects and patches supported libraries.

## Overview

All integrations are automatically enabled when available. No additional configuration is required beyond initializing ZeroEval:

```python
import zeroeval as ze
ze.init(api_key="YOUR_API_KEY")
```

## Supported Integrations

### 1. OpenAI

Automatically traces all OpenAI API calls including:

- Chat completions (streaming and non-streaming)
- Token usage tracking
- Input/output capture
- Error handling

**Traced Operations:**

- `client.chat.completions.create()`
- Streaming responses with automatic buffering

**Example:**

```python
import openai
client = openai.OpenAI()

# This call is automatically traced
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Note:** If you're using LiveKit Agents, disable this integration to prevent conflicts with LiveKit's OpenAI plugin. See the LiveKit section below.

### 2. LangChain

Comprehensive tracing for all LangChain components:

**Traced Components:**

- **Runnables**: `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`
- **LLMs**: All language model calls via `BaseLanguageModel`
- **Tools**: `BaseTool.run()` and `arun()`
- **Retrievers**: Document retrieval operations
- **Chains**: Sequential and parallel chain execution

**Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# All components are automatically traced
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
chain = prompt | model

# This creates a trace with spans for prompt + model
response = chain.invoke({"topic": "AI"})
```

### 3. LangGraph (Enhanced)

Our most comprehensive integration, tracing the full agentic workflow.

**Important:** You will see BOTH `langgraph.*` and `langchain.*` spans when using LangGraph:

- `langgraph.invoke`, `langgraph.stream` - High-level graph execution spans
- `langchain.*` - Individual node and component execution spans

This is expected behavior as LangGraph builds on top of LangChain components.

**Traced Operations:**

#### Graph Execution

- `invoke`, `ainvoke` - Full graph runs with metadata
- `stream`, `astream` - Streaming with node sequence tracking
- Graph structure metadata (nodes, edges, conditionals)

#### Node-Level Tracing

- **Individual node executions** - Each node gets its own span
- **State transformations** - Input/output state for each node
- **Execution timing** - Latency per node

#### Conditional Logic

- **Conditional edges** - Traces routing decisions
- **Dynamic flow** - Captures actual execution path

#### Tool Integration

- **Tool calls within nodes** - Integrated with LangChain tool tracing
- **Multi-step reasoning** - Full visibility into agent decision-making

#### Checkpointing (if enabled)

- **State persistence** - Save/load operations
- **Recovery points** - Checkpoint timing and size

**Enhanced Attributes:**

- `node_count` - Number of nodes in the graph
- `edge_count` - Number of edges
- `has_conditionals` - Whether graph has conditional routing
- `nodes` - List of node names
- `node_sequence` - Execution order during streaming
- `time_to_first_chunk` - Streaming latency metrics

**Example:**

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Define a multi-node graph
workflow = StateGraph(AgentState)
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add conditional routing
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)

app = workflow.compile()

# This creates a comprehensive trace hierarchy:
# - Parent span for full graph execution
# - Child spans for each node (reasoning, agent, tools)
# - Metadata about graph structure and routing
result = app.invoke({"messages": [HumanMessage(content="Help me plan a trip")]})
```

**Trace Hierarchy Example:**

```
langgraph.invoke (500ms)
├── langgraph.node.reasoning (50ms)
├── langgraph.node.agent (200ms)
├── langchain.invoke (180ms) [LLM call within agent]
├── langgraph.node.tools (150ms)
└── langchain.tool.run (140ms) [Tool execution]
```

### 4. LiveKit (Compatibility Mode)

**Current Status:** LiveKit compatibility mode prevents conflicts between ZeroEval's OpenAI instrumentation and LiveKit's OpenAI plugin.

**Important:** When using LiveKit Agents, you MUST disable the OpenAI integration:

```python
import zeroeval as ze

# REQUIRED: Disable OpenAI auto-patching for LiveKit compatibility
ze.init(
    api_key="YOUR_API_KEY",
    disabled_integrations=["openai"]
)
```

**Why is this necessary?**
LiveKit's OpenAI plugin uses a custom implementation that returns async context managers from its `chat()` method. ZeroEval's automatic OpenAI patching would interfere with this behavior, causing runtime errors.

**What still works:**

- All ZeroEval manual instrumentation (`@ze.span`, `ze.set_signal`, etc.)
- All other integrations (LangChain, LangGraph, etc.)
- Custom tracing and observability features

**Future Plans:**
We're working on a dedicated LiveKit integration that will properly instrument LiveKit's OpenAI calls without breaking compatibility. Stay tuned!

## Auto-Instrumentation Details

The ZeroEval tracer automatically:

1. **Detects installed packages** - Only patches libraries that are available
2. **Preserves functionality** - All original behavior is maintained
3. **Handles errors gracefully** - Tracing failures don't break your application
4. **Supports async operations** - Full async/await support
5. **Manages trace hierarchy** - Automatic parent-child span relationships

## Disabling Integrations

While integrations are automatic, you can disable specific ones if needed:

```python
import zeroeval as ze

# Method 1: Disable during initialization (recommended)
ze.init(
    api_key="YOUR_API_KEY",
    disabled_integrations=["openai", "langgraph"]  # Disable specific integrations
)

# Method 2: Via environment variable
# Set ZEROEVAL_DISABLED_INTEGRATIONS=openai,langgraph before running

# Method 3: Configure after initialization
from zeroeval.observability.tracer import tracer
tracer.configure(integrations={"openai": False, "langgraph": False})
```

**Common Use Cases for Disabling:**

- **LiveKit Users**: Disable `openai` to prevent conflicts with LiveKit's OpenAI plugin
- **Custom Instrumentation**: Disable auto-instrumentation when you have custom tracing
- **Performance**: Disable integrations you're not using to reduce overhead

## Performance Impact

All integrations are designed for minimal overhead:

- Trace data is buffered and sent asynchronously
- Sampling can be configured for high-volume applications
- Serialization happens outside the critical path

## Coming Soon

- **Anthropic** - Claude API tracing
- **Cohere** - Full Cohere platform support
- **HuggingFace** - Transformers and Inference API
- **LlamaIndex** - Document processing and retrieval
- **Custom Integrations** - SDK for building your own integrations

## Troubleshooting

If traces aren't appearing:

1. Check that ZeroEval is initialized before importing frameworks:

   ```python
   import zeroeval as ze
   ze.init()  # Must come first

   # Then import frameworks
   import openai
   import langchain
   ```

2. Verify integrations are loaded:

   ```python
   from zeroeval.observability.tracer import tracer
   print(tracer._integrations.keys())
   # Should show: ['OpenAIIntegration', 'LangChainIntegration', 'LangGraphIntegration']
   ```

3. Check for errors in initialization:
   ```bash
   export ZEROEVAL_LOG_LEVEL=DEBUG
   python your_script.py
   ```

For more help, contact us at [founders@zeroeval.com](mailto:founders@zeroeval.com)

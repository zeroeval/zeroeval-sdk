"""Advanced LangGraph Demo with Enhanced Tracing

This example demonstrates the expanded ZeroEval LangGraph integration that traces:
- Individual node executions
- Edge transitions
- Conditional routing
- State transformations
- Tool calls within nodes
- Streaming with node information
"""

import os
import asyncio
from typing import TypedDict, Annotated, Literal
from operator import add

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Initialize ZeroEval tracer before building graphs
import zeroeval as ze
from zeroeval.observability.tracer import tracer

# Configure tracer
tracer.configure(flush_interval=1.0, max_spans=100)
ze.init(api_key="sk_ze_3sGv8bqxdz0PShPu6-5c3TRlgPhD7w1QCnJB-bCzHpQ")

# Set up OpenAI API key
os.environ.setdefault("OPENAI_API_KEY", "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")

# -----------------------------------------------------------------------------
# Define the graph state
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    reasoning_steps: list[str]
    tool_calls_count: int
    should_continue: bool


# -----------------------------------------------------------------------------
# Define tools
# -----------------------------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated web search
    return f"Search results for '{query}': Found information about AI, machine learning, and {query}."


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"


tools = [search_web, calculate]

# -----------------------------------------------------------------------------
# Initialize model with tools
# -----------------------------------------------------------------------------
try:
    model = init_chat_model("openai:gpt-4o-mini").bind_tools(tools)
except:
    # Fallback if no API key
    model = None


# -----------------------------------------------------------------------------
# Define graph nodes
# -----------------------------------------------------------------------------
def reasoning_node(state: AgentState) -> AgentState:
    """Initial reasoning about the user's request."""
    last_message = state["messages"][-1]
    reasoning = f"Analyzing request: '{last_message.content}'"
    
    return {
        "reasoning_steps": [reasoning],
        "should_continue": True
    }


def agent_node(state: AgentState) -> AgentState:
    """Main agent that decides whether to use tools."""
    if model is None:
        # Fallback for demo without API key
        return {
            "messages": [AIMessage(content="I would use tools to help answer your question.")],
            "should_continue": False
        }
    
    # Call the model
    response = model.invoke(state["messages"])
    
    # Check if the model wants to use tools
    has_tool_calls = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
    
    return {
        "messages": [response],
        "tool_calls_count": state.get("tool_calls_count", 0) + (1 if has_tool_calls else 0),
        "should_continue": has_tool_calls
    }


def synthesize_node(state: AgentState) -> AgentState:
    """Synthesize final response from all information gathered."""
    tool_count = state.get("tool_calls_count", 0)
    synthesis = f"Synthesized response using {tool_count} tool calls."
    
    if tool_count == 0:
        # If no tools were used, we already have the final answer
        return state
    
    # Add a final synthesis message
    return {
        "messages": [AIMessage(content=synthesis)],
        "reasoning_steps": state["reasoning_steps"] + ["Final synthesis completed"]
    }


def should_continue(state: AgentState) -> Literal["tools", "synthesize"]:
    """Conditional edge that routes based on whether tools are needed."""
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return "synthesize"


# -----------------------------------------------------------------------------
# Build the graph
# -----------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("reasoning", reasoning_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("synthesize", synthesize_node)

# Add edges
workflow.add_edge(START, "reasoning")
workflow.add_edge("reasoning", "agent")

# Conditional routing from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "synthesize": "synthesize"
    }
)

# After tools, go back to agent
workflow.add_edge("tools", "agent")

# Synthesize leads to end
workflow.add_edge("synthesize", END)

# Compile the graph
app = workflow.compile()

# Optional: Add checkpointing for state persistence (will be traced)
# from langgraph.checkpoint.memory import MemorySaver
# app = workflow.compile(checkpointer=MemorySaver())


# -----------------------------------------------------------------------------
# Run examples showing different tracing scenarios
# -----------------------------------------------------------------------------
async def run_examples():
    print("🚀 LangGraph Advanced Tracing Demo\n")
    
    # Example 1: Simple question (no tools)
    print("=" * 60)
    print("Example 1: Simple question without tools")
    print("=" * 60)
    
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What is the capital of France?")]
    })
    
    print(f"Final response: {result['messages'][-1].content}")
    print(f"Tool calls: {result.get('tool_calls_count', 0)}")
    print()
    
    # Example 2: Question requiring tools
    print("=" * 60)
    print("Example 2: Question requiring tool usage")
    print("=" * 60)
    
    result = await app.ainvoke({
        "messages": [HumanMessage(content="Search for information about quantum computing and calculate 2^10")]
    })
    
    print(f"Final response: {result['messages'][-1].content}")
    print(f"Tool calls: {result.get('tool_calls_count', 0)}")
    print(f"Reasoning steps: {len(result.get('reasoning_steps', []))}")
    print()
    
    # Example 3: Streaming execution
    print("=" * 60)
    print("Example 3: Streaming with node visibility")
    print("=" * 60)
    
    print("Streaming nodes: ", end="", flush=True)
    async for event in app.astream(
        {"messages": [HumanMessage(content="What is 15 * 23?")]},
        stream_mode="values"
    ):
        # The enhanced integration will capture which nodes are executed
        node_executed = list(event.keys())
        if node_executed and node_executed[0] not in ["__root__", "messages"]:
            print(f"[{node_executed[0]}] ", end="", flush=True)
    
    print("\n\nDone! Check your ZeroEval dashboard to see:")
    print("- 🎯 Parent spans for each graph execution")
    print("- 📍 Individual spans for each node (reasoning, agent, tools, synthesize)")
    print("- 🔀 Conditional routing decisions")
    print("- 📊 Graph metadata (nodes, edges, conditionals)")
    print("- ⚡ Streaming information with node sequence")


if __name__ == "__main__":
    asyncio.run(run_examples()) 
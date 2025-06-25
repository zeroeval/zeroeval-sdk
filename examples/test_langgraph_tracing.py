"""Test LangGraph Tracing - Demonstrates the trace hierarchy"""

import time
import uuid
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Initialize ZeroEval tracer before building graphs
import zeroeval as ze
from zeroeval.observability.tracer import tracer
from zeroeval.observability.decorators import span

# Configure tracer
tracer.configure(flush_interval=1.0, max_spans=100)
ze.init(api_key="sk_ze_LIE7c4w3D7AasfKzzr-Qcff1Ts3R9d580HW5WjHf2HU")

# ---------------------------------------------------------------------------
# Global tags & session setup for this demo
# ---------------------------------------------------------------------------
# Project-level / example-level tags that we want on *every* span.
GLOBAL_TAGS = {
    "example": "test_langgraph_tracing",
    "project": "zeroeval",
}

# Create a shared session so all spans are grouped together.
SESSION_ID = str(uuid.uuid4())
SESSION_INFO = {"id": SESSION_ID, "name": "LangGraph Tracing Test"}

print("🔍 LangGraph Tracing Test")
print("=" * 60)
print("This test demonstrates the trace hierarchy created by LangGraph:")
print("- langgraph.* spans for graph-level operations")
print("- langchain.* spans for node executions and internal operations")
print("- Custom spans from @span decorators")
print("=" * 60)

# Simple state
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    counter: int

# Nodes with manual spans to show hierarchy
@span(
    name="custom.process_message",
    session=SESSION_INFO,
    tags={**GLOBAL_TAGS, "node": "process", "graph": "simple_demo", "env": "local", "team": "dev"},
)
def process_message(state: GraphState) -> GraphState:
    """First node - processes the message"""
    print("  → Processing message...")
    time.sleep(0.1)  # Simulate work
    
    with span(
        name="custom.extract_info",
        session=SESSION_INFO,
        tags={**GLOBAL_TAGS, "operation": "extract_info"},
    ):
        # Nested span to show hierarchy
        message_content = state["messages"][-1].content
        print(f"    Extracted: '{message_content}'")
    
    return {
        "messages": [AIMessage(content=f"Processed: {message_content}")],
        "counter": state.get("counter", 0) + 1
    }

@span(
    name="custom.enhance_message",
    session=SESSION_INFO,
    tags={**GLOBAL_TAGS, "node": "enhance", "graph": "simple_demo"},
)
def enhance_message(state: GraphState) -> GraphState:
    """Second node - enhances the processed message"""
    print("  → Enhancing message...")
    time.sleep(0.1)  # Simulate work
    
    last_msg = state["messages"][-1].content
    enhanced = f"{last_msg} [Enhanced with metadata]"
    
    return {
        "messages": [AIMessage(content=enhanced)],
        "counter": state.get("counter", 0) + 1
    }

def finalize_message(state: GraphState) -> GraphState:
    """Final node - no manual span to show contrast"""
    print("  → Finalizing...")
    return {
        "messages": [AIMessage(content="✅ Complete!")],
        "counter": state.get("counter", 0) + 1
    }

# Build the graph
print("\n📊 Building graph...")
workflow = StateGraph(GraphState)
workflow.add_node("process", process_message)
workflow.add_node("enhance", enhance_message)
workflow.add_node("finalize", finalize_message)

workflow.add_edge(START, "process")
workflow.add_edge("process", "enhance")
workflow.add_edge("enhance", "finalize")
workflow.add_edge("finalize", END)

app = workflow.compile()
print("✅ Graph compiled")

# Test 1: Invoke
print("\n🚀 Test 1: Graph Invocation")
print("-" * 40)
with span(
    name="test.invoke_workflow",
    session=SESSION_INFO,
    tags={**GLOBAL_TAGS, "operation": "invoke", "run_type": "invoke", "env": "local"},
):
    result = app.invoke({
        "messages": [HumanMessage(content="Hello, LangGraph!")],
        "counter": 0
    })

print(f"\nResult: {result['counter']} steps executed")
print(f"Final message: {result['messages'][-1].content}")

# Test 2: Stream
print("\n\n🚀 Test 2: Graph Streaming")
print("-" * 40)
with span(
    name="test.stream_workflow",
    session=SESSION_INFO,
    tags={**GLOBAL_TAGS, "operation": "stream"},
):
    print("Streaming events:")
    current_span = ze.get_current_span()
    ze.set_tag(current_span, {"unique": "stream_root"})

    for event in app.stream({"messages": [HumanMessage(content="Stream test")]}):
        node_name = list(event.keys())[0] if event else "unknown"
        if node_name not in ["__root__", "messages"]:
            print(f"  📍 Node executed: {node_name}")

print("\n\n✨ Trace Hierarchy Created:")
print("""
Expected trace structure:
└── test.invoke_workflow (manual span)
    └── langgraph.invoke (graph execution)
        ├── langchain.invoke (internal orchestration)
        ├── custom.process_message (node 1)
        │   └── custom.extract_info (nested span)
        ├── custom.enhance_message (node 2)
        └── langchain.invoke (node 3 - finalize)

Check your ZeroEval dashboard to see:
- 🎯 Graph-level spans (langgraph.*)
- 🔧 Node execution spans (custom.* and langchain.*)
- 📊 Performance metrics for each operation
- 🔄 State transformations between nodes
""")

# Flush traces
time.sleep(2)
tracer.flush()
print("\n✅ Traces flushed to ZeroEval dashboard") 
'''LangGraph Tags Demo

Demonstrates:
• Static tags via @span / with span blocks
• Automatic propagation to child spans
• Dynamic tagging of current span, trace and session using helper functions.
'''

import time
import uuid
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import zeroeval as ze
from zeroeval.observability.tracer import tracer
from zeroeval.observability.decorators import span

# Configure tracer for quick flushing in demos
tracer.configure(flush_interval=1.0, max_spans=100)
ze.init(api_key="sk_ze_LIE7c4w3D7AasfKzzr-Qcff1Ts3R9d580HW5WjHf2HU")

# Global demo-wide tags
DEMO_TAGS = {"example": "langgraph_tags_demo", "project": "zeroeval"}

# Shared session for all spans
SESSION_ID = str(uuid.uuid4())
SESSION_INFO = {"id": SESSION_ID, "name": "Tags Demo Session"}

# ---------------------- Graph definition -----------------------------
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    count: int

@span(
    name="demo.process",
    session=SESSION_INFO,
    tags={**DEMO_TAGS, "node": "process"},
)
def process(state: GraphState) -> GraphState:
    print("→ Processing message …")
    time.sleep(0.05)
    last = state["messages"][-1].content
    with span(
        name="demo.extract",
        session=SESSION_INFO,
        tags={**DEMO_TAGS, "operation": "extract"},
    ):
        extracted = last.upper()
    return {"messages": [AIMessage(content=extracted)], "count": state.get("count", 0) + 1}

@span(
    name="demo.enhance",
    session=SESSION_INFO,
    tags={**DEMO_TAGS, "node": "enhance"},
)
def enhance(state: GraphState) -> GraphState:
    print("→ Enhancing …")
    time.sleep(0.05)
    msg = state["messages"][-1].content + " ✨"
    return {"messages": [AIMessage(content=msg)], "count": state.get("count", 0) + 1}

def finish(state: GraphState) -> GraphState:
    print("→ Finishing …")
    return {"messages": [AIMessage(content="✅ done")], "count": state.get("count", 0) + 1}

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("process", process)
workflow.add_node("enhance", enhance)
workflow.add_node("finish", finish)
workflow.add_edge(START, "process")
workflow.add_edge("process", "enhance")
workflow.add_edge("enhance", "finish")
workflow.add_edge("finish", END)
app = workflow.compile()

# ------------------- Invocation with dynamic tags --------------------
print("\n🚀 Running graph with dynamic tag injection …")
with span(
    name="demo.root_invoke",
    session=SESSION_INFO,
    tags={**DEMO_TAGS, "run": "invoke"},
):
    root_span = ze.get_current_span()
    ze.set_tag(root_span, {"phase": "pre-run"})
    ze.set_tag(ze.get_current_trace(), {"run_mode": "invoke"})
    ze.set_tag(ze.get_current_session(), {"env": "local"})

    result = app.invoke({"messages": [HumanMessage(content="hello")], "count": 0})

print("Result message:", result["messages"][-1].content)

# ---------------- Streaming with further tag updates -----------------
print("\n🚀 Streaming run with additional tags …")
with span(
    name="demo.root_stream",
    session=SESSION_INFO,
    tags={**DEMO_TAGS, "run": "stream"},
):
    # Tag only this span
    ze.set_tag(ze.get_current_span(), {"unique": "stream_root"})

    for evt in app.stream({"messages": [HumanMessage(content="stream me")]}):
        node = list(evt.keys())[0]
        if node not in ["__root__", "messages"]:
            print("  • node", node)

# Flush everything to the backend
tracer.flush()
print("✅ demo finished – inspect tags in your ZeroEval dashboard") 
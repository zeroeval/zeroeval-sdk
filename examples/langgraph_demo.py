import os
import asyncio
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

# Initialise ZeroEval tracer (auto-sets up integrations) BEFORE building any
# LangGraph graphs so that their .compile() methods are patched.
import zeroeval as ze
from zeroeval.observability.tracer import tracer
from zeroeval.observability.decorators import span

# Common demo tags
GLOBAL_TAGS = {"demo": "langgraph", "project": "zeroeval"}

# -----------------------------------------------------------------------------
# SDK & tracer configuration (⚙️ tweak as needed)
# -----------------------------------------------------------------------------
tracer.configure(flush_interval=1.0, max_spans=50)

# Your ZeroEval API key – replace with a real one for production usage.
ze.init(api_key="sk_ze_V12rPuG8k2ee9uWg0p-wksm3b3h9tDgkR7nWZtxVy24")

# Generate a session ID for this run
session_id = str(uuid.uuid4())

# -----------------------------------------------------------------------------
# 1. Configure your LLM provider (OpenAI in this demo)
# -----------------------------------------------------------------------------
#   $ export OPENAI_API_KEY="sk-..."
#   $ pip install langgraph langchain-openai
# -----------------------------------------------------------------------------

os.environ.setdefault("OPENAI_API_KEY", "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")

# -----------------------------------------------------------------------------
# Helper – obtain an OpenAI key interactively (optional)
# -----------------------------------------------------------------------------


def _init_chat_model():
    """Return a ChatModel instance or None (echo fallback)."""

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        try:
            import getpass

            key = getpass.getpass(
                "OpenAI API key (press Enter to skip and use echo mode): "
            ).strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
        except Exception:
            # Non-interactive environment – fallback to echo.
            pass

    if key:
        return init_chat_model("openai:gpt-4o-mini")

    # No key – operate in echo mode.
    return None


# Lazy-init so the example still runs without a key.
CHAT_MODEL = _init_chat_model()


# -----------------------------------------------------------------------------
# 2. Build a minimal LangGraph workflow
# -----------------------------------------------------------------------------


@span(name="call_llm_node", session={"id": session_id, "name": "LangGraph Demo Session"},
      tags={**GLOBAL_TAGS, "node": "llm", "provider": "openai"})
def call_llm(state: MessagesState):
    """Simple node that calls an LLM (sync)."""
    if CHAT_MODEL is None:
        # Fallback to an echo response so the demo runs without a key.
        response = {
            "role": "assistant",
            "content": f"(echo) {state['messages'][-1]['content']}"
        }
    else:
        response = CHAT_MODEL.invoke(state["messages"])
    return {"messages": [response]}


@span(name="finish_node", session={"id": session_id, "name": "LangGraph Demo Session"},
      tags={**GLOBAL_TAGS, "node": "finish"})
def finish(state: MessagesState):
    """Terminal node – just returns the current state."""
    return {}


# Define the graph schema (just a list of messages)
GraphState = MessagesState

builder = StateGraph(GraphState)
builder.add_node("llm", call_llm)
builder.add_node("done", finish)

builder.add_edge(START, "llm")
builder.add_edge("llm", "done")
builder.add_edge("done", END)

app = builder.compile()

# -----------------------------------------------------------------------------
# 3. Run the graph – ZeroEval LangGraphIntegration will automatically trace it
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    user_input = input("Ask something: ") or "Hello LangGraph!"

    # Sync run ---------------------------------------------------------------
    @span(name="sync_invoke", session={"id": session_id, "name": "LangGraph Demo Session"},
          tags={**GLOBAL_TAGS, "operation": "sync_invoke"},
          trace_tags={"run_type": "sync"},
          session_tags={"env": "dev"})
    def run_sync():
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print("\nSync result:\n", result["messages"][-1].content)
        return result
    
    run_sync()

    # Async run --------------------------------------------------------------
    @span(name="async_invoke", session={"id": session_id, "name": "LangGraph Demo Session"},
          tags={**GLOBAL_TAGS, "operation": "async_invoke"},
          trace_tags={"run_type": "async"})
    async def async_run():
        out = await app.ainvoke({"messages": [HumanMessage(content="Async hi!")]})
        print("\nAsync result:\n", out["messages"][-1].content)

    asyncio.run(async_run())

    # Stream run -------------------------------------------------------------
    @span(name="stream_invoke", session={"id": session_id, "name": "LangGraph Demo Session"},
          tags={**GLOBAL_TAGS, "operation": "stream_invoke"},
          trace_tags={"run_type": "stream"})
    def run_stream():
        print("\nStreaming tokens:")
        for chunk in app.stream(
            {"messages": [HumanMessage(content="Stream please")]}, stream_mode="values"
        ):
            for msg in chunk.get("messages", []):
                if getattr(msg, "content", ""):
                    print(msg.content, end=" | ")
        print("\nDone.")
    
    run_stream() 
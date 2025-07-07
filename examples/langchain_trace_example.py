"""examples/langchain_trace_example.py

Demonstrates ZeroEval's automatic instrumentation of LangChain.
Run this file (`python examples/langchain_trace_example.py`) after you
have installed the extra dependencies:

```
poetry add langchain-openai  # or: pip install langchain-openai
```

Nothing in this script uses the ZeroEval tracer directly – spans are
created transparently by the LangChain integration we just added.
"""

from __future__ import annotations

import os
import time

import zeroeval as ze
from zeroeval.observability.tracer import tracer

# -----------------------------------------------------------------------------
# SDK & tracer configuration (⚙️ tweak as needed)
# -----------------------------------------------------------------------------
tracer.configure(flush_interval=1.0, max_spans=50)

# Your ZeroEval API key – replace with a real one for production usage.
ze.init(api_key="sk_ze_uGb9IzYU5gGxuEMpvo93DLObRbggfZz9g9eWjpzki4I")

# OpenAI key for the underlying LLM provider that LangChain will call.
os.environ.setdefault(
    "OPENAI_API_KEY",
    "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",
)

# -----------------------------------------------------------------------------
# LangChain imports
# -----------------------------------------------------------------------------
# 👇 Requires `langchain-openai` in addition to `langchain-core` (already in pyproject)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # type: ignore

# -----------------------------------------------------------------------------
# Build a simple Runnable chain: Prompt → Model → Parser
# -----------------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant."),
        ("human", "{question}"),
    ]
)

# Composition operator `|` creates a Runnable sequence.
chain = prompt | model | StrOutputParser()


# -----------------------------------------------------------------------------
# Main driver code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    q = "What is the capital of France?"

    print("Running chain.invoke() …")
    answer = chain.invoke({"question": q})
    print("Answer:", answer)

    print("\nRunning chain.stream() …")
    start = time.time()
    for chunk in chain.stream({"question": q}):
        print(chunk, end="", flush=True)
    print("\nstream elapsed:", round(time.time() - start, 2), "s")

    print("\n✅ Done! Check your ZeroEval dashboard for the new spans.")

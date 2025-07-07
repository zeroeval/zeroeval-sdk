"""examples/langchain_extended_trace_example.py

Showcase ZeroEval's *extended* LangChain integration added in May-2025.
We exercise the newly-instrumented abstractions:

• Tools – `BaseTool.run` / `arun`
• LLMs  – `BaseLanguageModel.generate` / `agenerate`
• Retrievers – `BaseRetriever.get_relevant_documents` / `__call__`
• Runnables – `__call__`, `invoke`, `stream`, … (already covered)

Spans for every call will appear in your ZeroEval dashboard – no manual
tracer usage required.

Usage
-----
$ poetry run python examples/langchain_extended_trace_example.py

Dependencies
------------
poetry add langchain-openai  # or: pip install langchain-openai

"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List

import zeroeval as ze
from zeroeval.observability.tracer import tracer

# -----------------------------------------------------------------------------
# SDK & tracer configuration (⚙️ tweak as needed)
# -----------------------------------------------------------------------------
tracer.configure(flush_interval=1.0, max_spans=100)

# Your ZeroEval API key – replace with a real one for production usage.
ze.init(api_key="sk_ze_uGb9IzYU5gGxuEMpvo93DLObRbggfZz9g9eWjpzki4I")

# OpenAI key for the underlying LLM provider that LangChain will call.
os.environ.setdefault(
    "OPENAI_API_KEY",
    "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",
)

# -----------------------------------------------------------------------------
# LangChain imports (requires `langchain-openai` extra)
# -----------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage  # type: ignore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI  # type: ignore

# -----------------------------------------------------------------------------
# 1️⃣  Custom Tool example
# -----------------------------------------------------------------------------


class EchoTool(BaseTool):
    """Very simple synchronous / asynchronous echo tool."""

    name: str = "echo"
    description: str = "Echoes back the given text."

    def _run(self, text: str) -> str:
        return f"Echo: {text}"

    async def _arun(self, text: str) -> str:
        return self._run(text)


echo_tool = EchoTool()

# -----------------------------------------------------------------------------
# 2️⃣  Custom Retriever example
# -----------------------------------------------------------------------------


class SimpleRetriever(BaseRetriever):
    """In-memory keyword retriever (sync + async)."""

    def __init__(self, docs: List[Document]):
        super().__init__()
        self._docs = docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return [d for d in self._docs if query.lower() in d.page_content.lower()]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Pretend to be costly 😉
        await asyncio.sleep(0.05)
        return self._get_relevant_documents(query)


documents = [
    Document(page_content="The Eiffel Tower is in Paris."),
    Document(page_content="London is the capital of the UK."),
    Document(page_content="Berlin is the capital of Germany."),
]
retriever = SimpleRetriever(documents)

# -----------------------------------------------------------------------------
# 3️⃣  LLM + Runnable chain example
# -----------------------------------------------------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly, concise assistant."),
        ("human", "{question}"),
    ]
)

chain = prompt | model | StrOutputParser()

# -----------------------------------------------------------------------------
# Helper coroutine to demo async paths
# -----------------------------------------------------------------------------


async def async_demo() -> None:
    print("\n🔹 EchoTool.arun() …")
    res = await echo_tool.arun("Hello async tool!")
    print(res)

    print("\n🔹 Retriever.aget_relevant_documents() …")
    docs = await retriever.aget_relevant_documents("capital")
    for d in docs:
        print("-", d.page_content)


# -----------------------------------------------------------------------------
# Main driver code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Tool (sync) ---------------------------------------------------------
    print("\n🔸 EchoTool.run() …")
    print(echo_tool.run("Hello tool!"))

    # --- Retriever (sync) ----------------------------------------------------
    print("\n🔸 Retriever.get_relevant_documents() …")
    for d in retriever.invoke("Paris"):
        print("-", d.page_content)

    # --- LLM generate() ------------------------------------------------------
    print("\n🔸 ChatOpenAI.generate() …")
    gen = model.generate([[HumanMessage(content="Name three US states.")]])
    print(gen.generations[0][0].text.strip())

    # --- Runnable __call__() -------------------------------------------------
    print("\n🔸 Runnable __call__() …")
    answer = chain.invoke({"question": "What is the capital of Germany?"})
    print("Answer:", answer)

    # --- Runnable stream() ---------------------------------------------------
    print("\n🔸 chain.stream() …")
    start = time.time()
    for chunk in chain.stream({"question": "Give me a short fun fact about Paris."}):
        print(chunk, end="", flush=True)
    print("\nstream elapsed:", round(time.time() - start, 2), "s")

    # --- Async paths ---------------------------------------------------------
    asyncio.run(async_demo())

    print("\n✅ Done! Check your ZeroEval dashboard for the full call graph.")

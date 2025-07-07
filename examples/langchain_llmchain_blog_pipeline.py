"""examples/langchain_llmchain_blog_pipeline.py

Advanced example: Multi-step content-creation pipeline using LangChain's LLMChain
and ZeroEval's automatic instrumentation.

What it does
------------
1. Generates a detailed outline for a blog post on a given topic.
2. Expands that outline into a full article (~600 words).

Both steps are implemented as `LLMChain`s and executed sequentially while
ZeroEval transparently records all spans (no explicit tracer calls needed).

Running the script
------------------
1. Install deps:
   ```bash
   # core SDK already installed – add extra deps for examples:
   poetry add langchain-openai  # or: pip install langchain-openai
   ```
2. Run it:
   ```bash
   python examples/langchain_llmchain_blog_pipeline.py
   ```

Environment variables
---------------------
• ZEROEVAL_API_KEY – optional; will fall back to the hard-coded demo key below.
• OPENAI_API_KEY    – required for the underlying LLM provider; falls back to a
  low-privilege demo key if unset (⚠️ replace in production!).
"""

from __future__ import annotations

import os

import zeroeval as ze
from zeroeval.observability.tracer import tracer

# -----------------------------------------------------------------------------
# SDK & tracer configuration (⚙️ tweak as needed)
# -----------------------------------------------------------------------------
tracer.configure(flush_interval=1.0, max_spans=50)

ze.init(
    api_key=os.getenv(
        "ZEROEVAL_API_KEY", "sk_ze_3sGv8bqxdz0PShPu6-5c3TRlgPhD7w1QCnJB-bCzHpQ"
    )
)

# OpenAI key for the underlying LLM provider that LangChain will call.
os.environ.setdefault(
    "OPENAI_API_KEY",
    "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",
)

# -----------------------------------------------------------------------------
# LangChain imports
# -----------------------------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # type: ignore

# -----------------------------------------------------------------------------
# Build two runnable chains (Prompt → Model → Parser)
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

outline_prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Generate a detailed outline (with bullet points) for a comprehensive "
        'blog post on the topic: "{topic}".\n\nOutline:\n'
    ),
)

article_prompt = PromptTemplate(
    input_variables=["topic", "outline"],
    template=(
        'Write a ~600-word blog post on the topic "{topic}" following the '
        "outline below. Ensure an engaging introduction, informative body, and "
        "concise conclusion.\n\nOutline:\n{outline}\n\nBlog post:\n"
    ),
)

# Composition operator builds pipelines that conform to the Runnable interface.
outline_chain = outline_prompt | llm | StrOutputParser()
article_chain = article_prompt | llm | StrOutputParser()

# -----------------------------------------------------------------------------
# Main driver code
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    topic = "How AI is transforming healthcare"

    print("\n📑 Generating outline …\n")
    outline_text: str = outline_chain.invoke({"topic": topic})
    print(outline_text)

    print("\n📝 Drafting article … (this may take a moment)\n")
    article_text: str = article_chain.invoke({"topic": topic, "outline": outline_text})
    print(article_text)

    print("\n✅ Done! Check your ZeroEval dashboard for the new spans.")

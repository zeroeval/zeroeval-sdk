"""
Real-World LangGraph Web Research Agent Example
==============================================

This example demonstrates a sophisticated AI research assistant that:
1. Performs intelligent web searches based on user queries
2. Decides when to search for more information vs. synthesize results
3. Summarizes findings from multiple sources
4. Handles follow-up questions and refinements
5. Shows comprehensive tracing of all agent actions

The agent uses a multi-step workflow:
- Query Analysis → Search Planning → Web Search → Result Evaluation → Synthesis
"""

import os
import asyncio
from typing import TypedDict, Annotated, List, Dict, Literal, Optional
from datetime import datetime
import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Initialize ZeroEval for comprehensive tracing
import zeroeval as ze
from zeroeval.observability.tracer import tracer
from zeroeval.observability.decorators import span

# Configure tracer for detailed agent tracing
tracer.configure(flush_interval=1.0, max_spans=200)
ze.init(api_key="sk_ze_uGb9IzYU5gGxuEMpvo93DLObRbggfZz9g9eWjpzki4I")

# Set up API keys
os.environ.setdefault("OPENAI_API_KEY", "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")


# -----------------------------------------------------------------------------
# Agent State Definition
# -----------------------------------------------------------------------------
class ResearchState(TypedDict):
    """State for the research agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    search_queries: List[str]
    search_results: List[Dict]
    sources: List[Dict]
    synthesis: Optional[str]
    requires_more_info: bool
    search_count: int
    max_searches: int


# -----------------------------------------------------------------------------
# Tool Definitions
# -----------------------------------------------------------------------------
@tool
def web_search(query: str, num_results: int = 5) -> Dict:
    """
    Search the web for information. Returns top results with snippets.
    In production, this would call a real API like Serper or Tavily.
    """
    # Simulated search results for demo
    mock_results = {
        "results": [
            {
                "title": f"Result 1 for: {query}",
                "url": f"https://example.com/1/{query.replace(' ', '-')}",
                "snippet": f"This is a detailed article about {query}. It covers various aspects including implementation details, best practices, and real-world applications.",
                "date": "2024-01-15"
            },
            {
                "title": f"Research Paper: {query}",
                "url": f"https://arxiv.org/2/{query.replace(' ', '-')}",
                "snippet": f"Academic research on {query} showing latest developments and theoretical foundations. Includes empirical studies and benchmarks.",
                "date": "2024-02-20"
            },
            {
                "title": f"Tutorial: Getting Started with {query}",
                "url": f"https://tutorial.com/3/{query.replace(' ', '-')}",
                "snippet": f"Step-by-step guide for beginners on {query}. Learn the fundamentals and build your first project.",
                "date": "2023-12-10"
            }
        ],
        "query": query,
        "search_time": 0.234
    }
    
    return mock_results


@tool
def extract_key_facts(text: str) -> List[str]:
    """Extract key facts from a piece of text."""
    # In production, this might use NLP or another LLM call
    facts = [
        f"Key finding from text: {text[:50]}...",
        "Important consideration mentioned in the source",
        "Practical application identified"
    ]
    return facts


# Initialize tools
tools = [web_search, extract_key_facts]

# Initialize model
try:
    model = init_chat_model("openai:gpt-4o-mini")
    search_model = model.bind_tools(tools)
except:
    print("No OpenAI key found. Using mock model.")
    model = None
    search_model = None


# -----------------------------------------------------------------------------
# Node Definitions
# -----------------------------------------------------------------------------
@span(name="agent.analyze_query")
def analyze_query_node(state: ResearchState) -> ResearchState:
    """Analyze the user query and plan search strategy."""
    query = state["messages"][-1].content if state["messages"] else state.get("query", "")
    
    # Analyze complexity and create search queries
    search_queries = [
        query,  # Original query
        f"{query} latest developments 2024",  # Recent info
        f"{query} best practices",  # Practical info
    ]
    
    return {
        "query": query,
        "search_queries": search_queries[:2],  # Start with first 2
        "requires_more_info": True,
        "search_count": 0,
        "max_searches": 3
    }


@span(name="agent.search_web") 
def search_node(state: ResearchState) -> ResearchState:
    """Execute web searches based on the search plan."""
    search_results = state.get("search_results", [])
    search_count = state.get("search_count", 0)
    
    # Get next search query
    remaining_queries = [q for q in state["search_queries"] 
                        if not any(r.get("query") == q for r in search_results)]
    
    if not remaining_queries:
        return {"requires_more_info": False}
    
    next_query = remaining_queries[0]
    print(f"\n🔍 Searching for: '{next_query}'")
    
    # Perform search
    if search_model:
        # Real search using tool
        result = search_model.invoke([
            SystemMessage(content="You are a research assistant. Search for the following query."),
            HumanMessage(content=f"Search for: {next_query}")
        ])
        # Extract tool call results
        search_data = {"query": next_query, "results": []}  # Would parse from tool_calls
    else:
        # Mock search
        search_data = web_search.invoke({"query": next_query})
    
    search_results.append(search_data)
    search_count += 1
    
    return {
        "search_results": search_results,
        "search_count": search_count,
        "messages": [AIMessage(content=f"Completed search {search_count} of {state['max_searches']}")],
        "requires_more_info": search_count < state.get("max_searches", 3)
    }


@span(name="agent.evaluate_results")
def evaluate_results_node(state: ResearchState) -> ResearchState:
    """Evaluate search results and extract relevant information."""
    all_sources = []
    
    for search in state.get("search_results", []):
        for result in search.get("results", []):
            source = {
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "relevance_score": 0.85,  # Would be calculated
                "key_facts": extract_key_facts.invoke({"text": result["snippet"]})
            }
            all_sources.append(source)
    
    # Sort by relevance
    all_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "sources": all_sources[:10],  # Top 10 sources
        "messages": [AIMessage(content=f"Evaluated {len(all_sources)} sources")]
    }


@span(name="agent.synthesize")
def synthesize_node(state: ResearchState) -> ResearchState:
    """Synthesize findings into a comprehensive answer."""
    query = state.get("query", "")
    sources = state.get("sources", [])
    
    if not sources:
        synthesis = f"I couldn't find sufficient information about '{query}'."
    else:
        # Create synthesis (in production, would use LLM)
        synthesis = f"""
Based on my research on "{query}", here's what I found:

## Key Findings

"""
        for i, source in enumerate(sources[:3], 1):
            synthesis += f"{i}. **{source['title']}**\n"
            synthesis += f"   - {source['snippet']}\n"
            synthesis += f"   - Source: {source['url']}\n\n"
        
        synthesis += f"""
## Summary

I analyzed {len(sources)} sources and found comprehensive information about {query}. 
The sources include recent developments, best practices, and practical applications.

Would you like me to dive deeper into any specific aspect?
"""
    
    return {
        "synthesis": synthesis,
        "messages": [AIMessage(content=synthesis)],
        "requires_more_info": False
    }


def should_search_more(state: ResearchState) -> Literal["search", "evaluate", "synthesize"]:
    """Decide whether to search more or move to synthesis."""
    search_count = state.get("search_count", 0)
    max_searches = state.get("max_searches", 3)
    requires_more = state.get("requires_more_info", True)
    
    if search_count == 0:
        return "search"
    elif requires_more and search_count < max_searches:
        return "search"
    elif state.get("search_results"):
        return "evaluate"
    else:
        return "synthesize"


# -----------------------------------------------------------------------------
# Build the Graph
# -----------------------------------------------------------------------------
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("analyze", analyze_query_node)
workflow.add_node("search", search_node)
workflow.add_node("evaluate", evaluate_results_node)
workflow.add_node("synthesize", synthesize_node)

# Add edges
workflow.add_edge(START, "analyze")

# Conditional routing after analysis
workflow.add_conditional_edges(
    "analyze",
    should_search_more,
    {
        "search": "search",
        "evaluate": "evaluate",
        "synthesize": "synthesize"
    }
)

# After search, decide next step
workflow.add_conditional_edges(
    "search",
    should_search_more,
    {
        "search": "search",
        "evaluate": "evaluate", 
        "synthesize": "synthesize"
    }
)

# Evaluate leads to synthesis
workflow.add_edge("evaluate", "synthesize")

# Synthesis leads to end
workflow.add_edge("synthesize", END)

# Compile with memory for conversation support
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
async def research_demo():
    """Demonstrate the research agent with different queries."""
    
    print("🤖 AI Research Assistant")
    print("=" * 60)
    
    # Example 1: Technical Research
    print("\n📚 Example 1: Technical Research Query")
    print("-" * 60)
    
    config = {"configurable": {"thread_id": "research-1"}}
    
    result = await app.ainvoke({
        "messages": [HumanMessage(content="What are the latest developments in quantum computing error correction?")],
        "max_searches": 3
    }, config=config)
    
    print(f"\n✅ Research completed!")
    print(f"- Searches performed: {result.get('search_count', 0)}")
    print(f"- Sources found: {len(result.get('sources', []))}")
    print(f"\nSynthesis preview: {result.get('synthesis', '')[:200]}...")
    
    # Example 2: Follow-up Question
    print("\n\n📚 Example 2: Follow-up Question")
    print("-" * 60)
    
    result2 = await app.ainvoke({
        "messages": [HumanMessage(content="Can you explain more about surface codes specifically?")],
    }, config=config)
    
    print(f"\nFollow-up handled with context from previous search")
    
    # Example 3: Streaming Research
    print("\n\n📚 Example 3: Streaming Research Progress")
    print("-" * 60)
    
    print("Researching 'How do large language models handle context windows?'")
    print("\nProgress: ", end="", flush=True)
    
    config3 = {"configurable": {"thread_id": "research-3"}}
    
    async for event in app.astream({
        "messages": [HumanMessage(content="How do large language models handle context windows?")],
        "max_searches": 2
    }, config=config3):
        # Show which node is executing
        for key in event:
            if key not in ["__root__", "messages"]:
                print(f"[{key}]", end=" ", flush=True)
    
    print("\n\n✨ Done! Check your ZeroEval dashboard to see:")
    print("- 🎯 Complete agent execution traces")
    print("- 🔍 Individual search operations") 
    print("- 🧠 Decision points and routing")
    print("- 📊 Performance metrics for each step")
    print("- 🔄 State transformations throughout the workflow")


if __name__ == "__main__":
    asyncio.run(research_demo()) 
#!/usr/bin/env python3
"""
Customer Support Agent with PydanticAI + ZeroEval Prompt Management
===================================================================

This example demonstrates how to build a customer support agent using PydanticAI
with ZeroEval's `ze.prompt()` for prompt management and versioning.

Key concepts:
1. `ze.prompt()`: Manages prompts with variable interpolation and versioning
2. PydanticAI Agent with typed output
3. Tool definitions for agent actions
4. Streaming iteration through agent nodes
5. ZeroEval automatic tracing
"""

import asyncio
import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Load environment variables BEFORE importing zeroeval
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import zeroeval as ze


# =============================================================================
# Pydantic Models
# =============================================================================

class ActionType(str, Enum):
    """Types of actions the agent can suggest."""
    NONE = "none"
    CHECK_ORDER = "check_order"
    INITIATE_RETURN = "initiate_return"
    APPLY_DISCOUNT = "apply_discount"
    ESCALATE = "escalate"
    RECOMMEND_BOOKS = "recommend_books"


class SuggestedAction(BaseModel):
    """An action the agent suggests taking."""
    action_type: ActionType = Field(description="The type of action to take")
    details: Optional[str] = Field(default=None, description="Additional details about the action")


class AgentOutput(BaseModel):
    """Structured output from the customer support agent."""
    message: str = Field(description="The response message to show the customer")
    sentiment: str = Field(description="Detected customer sentiment: positive, neutral, negative, or frustrated")
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence in the response (0-1)")
    suggested_action: SuggestedAction = Field(description="Suggested follow-up action")
    requires_human_review: bool = Field(description="Whether this conversation should be flagged for human review")


class UserContext(BaseModel):
    """Customer context information passed as dependencies."""
    name: str = Field(default="there")
    membership: str = Field(default="Standard")
    user_id: Optional[str] = Field(default=None)


# =============================================================================
# Chunk Models for Streaming Output
# =============================================================================

class BaseChunk(BaseModel):
    """Base class for streaming chunks."""
    id: str
    kind: str
    created_at: datetime


class UserPromptChunk(BaseChunk):
    kind: str = "user_prompt"
    content: str


class ToolCallChunk(BaseChunk):
    kind: str = "tool_call"
    tool_name: str
    args: str


class ToolResultChunk(BaseChunk):
    kind: str = "tool_result"
    tool_name: str
    content: str
    metadata: Optional[dict] = None


class TextChunk(BaseChunk):
    kind: str = "text_result"
    content: str


class ModelOutputChunk(BaseChunk):
    kind: str = "model_output"
    content: str


# =============================================================================
# Agent Setup
# =============================================================================

# Initialize ZeroEval
ze.init(
    api_key=os.getenv("ZEROEVAL_API_KEY"),
    api_url=os.getenv("ZEROEVAL_API_URL", "http://localhost:8000")
)

# Prompt template with {{variable}} syntax for ze.prompt()
SYSTEM_PROMPT_TEMPLATE = """You are Elena, a customer support specialist at Bibliophile Books.

Customer Information:
- Name: {{user_name}}
- Membership: {{membership}}

Your task is to:
1. Respond helpfully to the customer's query
2. Detect their emotional sentiment
3. Suggest appropriate follow-up actions
4. Flag conversations that need human review (complaints, refund requests over $50, etc.)

Guidelines for Gold members: Free shipping, priority support, 15% off all purchases.
Guidelines for Standard members: Mention Gold benefits when relevant.

Be warm, personable, and concise. Use the customer's name naturally.

You have access to tools to check orders, recommend books, and apply discounts."""


def create_agent(user_context: UserContext) -> Agent[UserContext, AgentOutput]:
    """Create a PydanticAI agent for customer support.
    
    Uses ze.prompt() to register and manage the system prompt with ZeroEval.
    This enables prompt versioning, A/B testing, and optimization.
    """
    
    # Use ze.prompt() for prompt management and variable interpolation
    # This registers the prompt with ZeroEval and allows for versioning
    system_prompt = ze.prompt(
        name="bookstore-support-pydantic-agent",
        content=SYSTEM_PROMPT_TEMPLATE,
        variables={
            "user_name": user_context.name,
            "membership": user_context.membership
        }
    )
    
    agent = Agent(
        model="openai:gpt-4o-mini",
        output_type=AgentOutput,
        system_prompt=system_prompt,
        deps_type=UserContext,
    )
    
    # Define tools
    @agent.tool
    async def check_order_status(ctx, order_id: str) -> str:
        """Check the status of a customer's order."""
        # Simulated order lookup
        return f"Order {order_id}: Shipped on Nov 25, expected delivery Nov 28."
    
    @agent.tool
    async def recommend_books(ctx, genre: str, limit: int = 3) -> str:
        """Recommend books based on genre preference."""
        recommendations = {
            "fiction": ["The Midnight Library", "Project Hail Mary", "Klara and the Sun"],
            "mystery": ["The Thursday Murder Club", "The Maid", "The Paris Apartment"],
            "fantasy": ["The House in the Cerulean Sea", "Piranesi", "The Invisible Life of Addie LaRue"],
        }
        books = recommendations.get(genre.lower(), ["The Alchemist", "Educated", "Atomic Habits"])
        return f"Recommended {genre} books: {', '.join(books[:limit])}"
    
    @agent.tool
    async def apply_discount(ctx, discount_code: str) -> str:
        """Apply a discount code to the customer's account."""
        if ctx.deps.membership == "Gold":
            return f"Discount code '{discount_code}' applied! Combined with your Gold member 15% discount."
        return f"Discount code '{discount_code}' applied to your account."
    
    return agent


def _get_content(content) -> str:
    """Extract string content from various types."""
    if isinstance(content, str):
        return content
    elif hasattr(content, 'model_dump_json'):
        return content.model_dump_json()
    elif hasattr(content, '__str__'):
        return str(content)
    return repr(content)


async def run_agent_with_streaming(
    agent: Agent,
    user_input: str,
    user_context: UserContext,
    message_history: list | None = None
):
    """
    Run the agent and yield streaming chunks for each node.
    
    This demonstrates the agent.iter() pattern for processing
    agent execution step-by-step.
    """
    if message_history is None:
        message_history = []
    
    async with agent.iter(user_input, deps=user_context, message_history=message_history) as agent_run:
        async for node in agent_run:
            chunk = None
            
            if Agent.is_user_prompt_node(node):
                chunk = UserPromptChunk(
                    id=str(uuid4()),
                    kind="user_prompt",
                    content=node.user_prompt,
                    created_at=datetime.now(UTC)
                )
                yield chunk
                
            elif Agent.is_call_tools_node(node):
                for part in node.model_response.parts:
                    if isinstance(part, ToolCallPart):
                        chunk = ToolCallChunk(
                            id=part.tool_call_id or str(uuid4()),
                            kind="tool_call",
                            tool_name=part.tool_name,
                            args=part.args_as_json_str(),
                            created_at=datetime.now(UTC)
                        )
                        yield chunk
                        
            elif Agent.is_model_request_node(node):
                for part in node.request.parts:
                    if isinstance(part, ToolReturnPart):
                        chunk = ToolResultChunk(
                            id=part.tool_call_id or str(uuid4()),
                            kind="tool_result",
                            tool_name=part.tool_name,
                            content=_get_content(part.content),
                            metadata=None,
                            created_at=datetime.now(UTC)
                        )
                        yield chunk
                    elif isinstance(part, TextPart):
                        chunk = TextChunk(
                            id=str(uuid4()),
                            kind="text_result",
                            content=_get_content(part.content),
                            created_at=datetime.now(UTC)
                        )
                        yield chunk
                    elif isinstance(part, UserPromptPart):
                        pass  # Already handled
                        
            elif Agent.is_end_node(node):
                output = node.data.output
                chunk = ModelOutputChunk(
                    id=str(uuid4()),
                    kind="model_output",
                    content=_get_content(output),
                    created_at=datetime.now(UTC)
                )
                yield chunk
                
                # Return the final result and updated message history
                yield {
                    "final_output": output,
                    "message_history": agent_run.result.all_messages()
                }


async def main():
    print("\n=== Bookstore Support Agent (PydanticAI) ===")
    print("Type 'exit' to quit\n")
    
    user_context = UserContext(name="Alice", membership="Gold", user_id="user_123")
    print(f"Context: User={user_context.name}, Membership={user_context.membership}\n")
    
    # Create the agent (uses ze.prompt() internally for prompt management)
    print("Initializing agent with ze.prompt()...")
    agent = create_agent(user_context)
    print("Agent created with prompt: 'bookstore-support-pydantic-agent'\n")
    
    message_history = []
    
    # Initial greeting
    intro_query = "Hello! Please introduce yourself and tell me what you can help me with."
    print(f"You: {intro_query}\n")
    
    print("--- Agent Processing ---")
    final_output = None
    
    async for chunk in run_agent_with_streaming(agent, intro_query, user_context, message_history):
        if isinstance(chunk, dict) and "final_output" in chunk:
            final_output = chunk["final_output"]
            message_history = chunk["message_history"]
        elif isinstance(chunk, BaseChunk):
            print(f"  [{chunk.kind}] {chunk.id[:8]}...")
    
    if final_output:
        print(f"\nElena: {final_output.message}")
        print(f"  [Sentiment: {final_output.sentiment} | Confidence: {final_output.confidence:.0%}]")
        print(f"  [Action: {final_output.suggested_action.action_type.value}]")
        if final_output.requires_human_review:
            print("  [⚠️ Flagged for human review]")
    print()
    
    # Interactive loop
    while True:
        try:
            user_query = input("You: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            
            print("\n--- Agent Processing ---")
            final_output = None
            
            async for chunk in run_agent_with_streaming(agent, user_query, user_context, message_history):
                if isinstance(chunk, dict) and "final_output" in chunk:
                    final_output = chunk["final_output"]
                    message_history = chunk["message_history"]
                elif isinstance(chunk, BaseChunk):
                    print(f"  [{chunk.kind}] {chunk.id[:8]}...")
            
            if final_output:
                print(f"\nElena: {final_output.message}")
                print(f"  [Sentiment: {final_output.sentiment} | Confidence: {final_output.confidence:.0%}]")
                print(f"  [Action: {final_output.suggested_action.action_type.value}]")
                if final_output.suggested_action.details:
                    print(f"  [Details: {final_output.suggested_action.details}]")
                if final_output.requires_human_review:
                    print("  [⚠️ Flagged for human review]")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    asyncio.run(main())

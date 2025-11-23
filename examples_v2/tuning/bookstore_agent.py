#!/usr/bin/env python3
"""
Customer Support Agent with Tuning
=================================

This example demonstrates how to build a customer support agent using ZeroEval's
tuning features. It uses `ze.prompt()` to manage the prompt and `ze.send_feedback()`
to provide signals for optimization.

Key concepts:
1. `ze.prompt()`: Defines the prompt and binds variables for interpolation
2. Automatic Tracing: The SDK automatically traces OpenAI calls
3. Interactive Mode: You can chat with the agent and see how it responds
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing zeroeval
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import openai
import zeroeval as ze

# 1. Initialize ZeroEval
# Ensure you have ZEROEVAL_API_KEY and ZEROEVAL_API_URL set in your environment
ze.init(
    api_key=os.getenv("ZEROEVAL_API_KEY"),
    api_url=os.getenv("ZEROEVAL_API_URL", "http://localhost:8000"),
)

def customer_support_agent(user_query: str, user_context: dict = None, conversation_history: list = None):
    """
    A simple customer support agent that uses a managed prompt and maintains conversation history.
    """
    if user_context is None:
        user_context = {}
    if conversation_history is None:
        conversation_history = []

    # 2. Define the prompt using ze.prompt()
    # This registers the prompt with ZeroEval (if not exists) and allows for versioning.
    # The 'content' is your base prompt. You can use {{variable}} syntax.
    # 'variables' are passed for interpolation and tracking.
    
    prompt_name = "bookstore-support-agent"
    
    system_instruction = ze.prompt(
        name=prompt_name,
        content="""You are Elena, a passionate book enthusiast and customer support specialist at Bibliophile Books. You've worked in the bookstore for 5 years and genuinely love helping people discover their next great read.

Your personality:
- Warm and personable, like chatting with a knowledgeable friend at a bookshop
- Enthusiastic about books and reading
- Patient and empathetic when customers have issues
- Professional but not overly formal
- You use the customer's name naturally in conversation

Customer Information:
- Name: {{user_name}}
- Membership Level: {{membership}}

Guidelines:
1. Address {{user_name}} directly and warmly (but don't say "Hi {{user_name}}" in every message if you're in an ongoing conversation)
2. For Gold members: Remember they have free shipping, priority support, and 15% off all purchases
3. For Standard members: Offer helpful service while mentioning Gold membership benefits when relevant
4. Keep responses concise but friendly (2-4 sentences for simple queries)
5. If you don't know something or can't help, offer to connect them with a specialist
6. Never use placeholder text like "[Your Name]" - you are Elena
7. End naturally without formal sign-offs unless it's clearly the end of the conversation
8. IMPORTANT: Remember information from the conversation history and don't ask for things the customer already told you

Respond directly to their query in a helpful, personable way.""",
        variables={
            "user_name": user_context.get("name", "there"),
            "membership": user_context.get("membership", "Standard")
        }
    )

    # Initialize OpenAI client (ZeroEval automatically instruments this)
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\n--- Sending Request to AI ({prompt_name}) ---")
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_query})
    
    # 3. Call the Model
    # The SDK intercepts this call:
    # - Detects the <zeroeval> metadata from ze.prompt()
    # - Interpolates variables into the content
    # - Traces the execution
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use a cost-effective model
        messages=messages,
        temperature=0.7
    )

    completion_text = response.choices[0].message.content
    completion_id = response.id
    
    return completion_text, completion_id, prompt_name

def main():
    # Example interaction
    print("\n=== Bookstore Support Agent (Type 'exit' to quit) ===")
    
    # We'll assume a fixed user context for this session
    user_context = {
        "name": "Alice",
        "membership": "Gold" # VIP customer
    }
    print(f"Context: User={user_context['name']}, Membership={user_context['membership']}\n")
    
    # Initialize conversation history
    conversation_history = []
    
    # Agent introduces itself
    intro_query = "Hello! Please introduce yourself and ask how you can help me today."
    response_text, _, _ = customer_support_agent(intro_query, user_context, conversation_history)
    print(f"Elena: {response_text}\n")
    
    # Add intro to history
    conversation_history.append({"role": "user", "content": intro_query})
    conversation_history.append({"role": "assistant", "content": response_text})
    
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            response_text, completion_id, prompt_slug = customer_support_agent(user_query, user_context, conversation_history)
            
            print(f"\nElena: {response_text}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_query})
            conversation_history.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Check your ZEROEVAL_API_KEY and OPENAI_API_KEY.")
            break

if __name__ == "__main__":
    main()

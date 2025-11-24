#!/usr/bin/env python3
"""
Customer Support Agent with API Feedback Loop
===================================================

This example demonstrates how to submit feedback using the ZeroEval API directly,
bypassing the SDK's `ze.send_feedback` helper. This is useful for:
1. Frontend applications calling the backend directly
2. Systems where the SDK is not installed
3. Custom integrations

Key concepts:
- `POST /v1/prompts/{slug}/completions/{id}/feedback`: The feedback endpoint
- Direct API interaction
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE importing zeroeval
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import openai
import zeroeval as ze

# Configuration
API_URL = os.getenv("ZEROEVAL_API_URL", "http://localhost:8000")
API_KEY = os.getenv("ZEROEVAL_API_KEY")  # Use your ZeroEval API Key

# 1. Initialize ZeroEval
ze.init(
    api_key=API_KEY,
    api_url=API_URL,
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def customer_support_agent(user_query: str, user_context: dict = None, conversation_history: list = None):
    """
    A simple customer support agent that uses a managed prompt and maintains conversation history.
    """
    if user_context is None:
        user_context = {}
    if conversation_history is None:
        conversation_history = []

    # 2. Define the prompt using ze.prompt()
    prompt_name = "bookstore-support-agent-with-api-feedback"
    
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
- Address {{user_name}} directly and warmly
- For Gold members: Remember they have free shipping, priority support, and 15% off all purchases
- For Standard members: Offer helpful service while mentioning Gold membership benefits when relevant
- Keep responses concise but friendly
- If you don't know something or can't help, offer to connect them with a specialist
- Never use placeholder text like "[Your Name]" - you are Elena

Respond directly to their query in a helpful, personable way.""",
        variables={
            "user_name": user_context.get("name", "there"),
            "membership": user_context.get("membership", "Standard")
        }
    )

    print(f"\n--- Sending Request to AI ({prompt_name}) ---")
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": system_instruction}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_query})
    
    # 3. Call the Model
    # The SDK intercepts this call and tracks the completion_id
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use a cost-effective model for the agent
        messages=messages,
        temperature=0.7
    )

    completion_text = response.choices[0].message.content
    completion_id = response.id
    
    return completion_text, completion_id, prompt_name

def evaluate_response(user_query: str, agent_response: str):
    """
    Uses a powerful model (Evaluator) to grade the agent's response.
    Returns (is_good: bool, reason: str)
    """
    print("\n--- Running Evaluator (GPT-4o) ---")
    
    eval_prompt = f"""You are an expert customer support quality assurance specialist. 
    Your job is to evaluate a customer support response.

    User Query: "{user_query}"
    Agent Response: "{agent_response}"

    Criteria:
    1. Is the tone warm and professional?
    2. Is the information accurate and helpful?
    3. Does it address the user's specific query?

    Output strictly in JSON format with these fields:
    - "score": 1 to 5 (5 being perfect)
    - "reason": A brief explanation of the score
    - "thumbs_up": true if score >= 4, else false
    """

    response = client.chat.completions.create(
        model="gpt-4o", # Use a powerful model for evaluation
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error parsing evaluation: {e}")
        return {"thumbs_up": True, "reason": "Failed to parse evaluation", "score": 5}

def send_feedback_via_api(prompt_slug, completion_id, thumbs_up, reason=None, expected_output=None, metadata=None):
    """
    Sends feedback directly using requests.post to the ZeroEval API.
    """
    url = f"{API_URL}/v1/prompts/{prompt_slug}/completions/{completion_id}/feedback"
    
    payload = {
        "thumbs_up": thumbs_up,
        "reason": reason,
        "expected_output": expected_output,
        "metadata": metadata or {}
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"\n[API] POST {url}")
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        print("✓ API Feedback submitted successfully")
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"❌ API Request failed: {e}")
        print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"❌ Error sending feedback: {e}")
        return None

def main():
    # Example interaction
    print("\n=== Bookstore Support Agent with API Feedback (Type 'exit' to quit) ===")
    
    user_context = {
        "name": "Alice",
        "membership": "Gold" # VIP customer
    }
    print(f"Context: User={user_context['name']}, Membership={user_context['membership']}\n")
    
    conversation_history = []
    
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
            if not user_query:
                continue
                
            if user_query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            # 1. Get response from the agent
            response_text, completion_id, prompt_slug = customer_support_agent(
                user_query, 
                user_context, 
                conversation_history
            )
            
            print(f"\nElena: {response_text}")
            print(f"\n[DEBUG] OpenAI completion_id: {completion_id}")
            print(f"[DEBUG] Prompt slug: {prompt_slug}")
            
            # 2. Generate feedback using a powerful model
            # In a real system, this might happen asynchronously or be sampled
            eval_result = evaluate_response(user_query, response_text)
            
            print(f"\n[Evaluator] Score: {eval_result.get('score')}/5")
            print(f"[Evaluator] Reason: {eval_result.get('reason')}")
            print(f"[Evaluator] Verdict: {'👍 Thumbs Up' if eval_result.get('thumbs_up') else '👎 Thumbs Down'}")
            
            # 3. Submit feedback via API directly
            send_feedback_via_api(
                prompt_slug=prompt_slug,
                completion_id=completion_id,
                thumbs_up=eval_result.get("thumbs_up", True),
                reason=eval_result.get("reason"),
                metadata={
                    "score": eval_result.get("score"),
                    "evaluator_model": "gpt-4o",
                    "source": "direct_api"
                }
            )
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_query})
            conversation_history.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()


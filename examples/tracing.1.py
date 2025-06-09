import zeroeval as ze
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import openai
import numpy as np
import os
import json
import time
import uuid

# --- Initialize ZeroEval ---
# This will automatically patch the OpenAI client for tracing
# Make sure to set your ZEROEVAL_API_KEY environment variable
ze.init(api_key="sk_ze_f7mb9PQNbQEfOVSurY4S29B9YiUwrvO96Vi6QeicThU")


# --- Configure Tracer ---
# For this demo, we'll flush more frequently to see results quickly
tracer.configure(flush_interval=5.0)

# --- OpenAI Client ---
# The integration will automatically trace calls made with this client
# Make sure to set your OPENAI_API_KEY environment variable
try:
    client = openai.OpenAI(api_key="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")
except openai.OpenAIError as e:
    print("Error: OpenAI API key not configured.")
    print("Please set the OPENAI_API_KEY environment variable.")
    exit(1)


# --- Tool Functions ---
# These are the functions our agent can "call".

@span(name="tool.calculate_dot_product")
def calculate_dot_product(vector_a: list, vector_b: list) -> float:
    """Calculates the dot product of two vectors."""
    try:
        a = np.array(vector_a, dtype=float)
        b = np.array(vector_b, dtype=float)
        result = np.dot(a, b)
        return float(result)
    except Exception as e:
        return f"Error calculating dot product: {e}"

@span(name="tool.get_weather_forecast")
def get_weather_forecast(location: str, unit: str = "celsius") -> str:
    """Gets the weather forecast for a given location."""
    # This is a mock function for demonstration
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "15", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

# --- Agent Logic ---

available_tools = {
    "calculate_dot_product": calculate_dot_product,
    "get_weather_forecast": get_weather_forecast,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "calculate_dot_product",
            "description": "Calculates the dot product of two numerical vectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vector_a": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The first vector."
                    },
                    "vector_b": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The second vector."
                    }
                },
                "required": ["vector_a", "vector_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]

def run_conversation_turn(session_id: str, messages: list):
    """
    Runs a single turn of the conversation, which constitutes a single trace
    within the broader session.
    """
    with span(name="agent.interaction", session_id=session_id) as current_span:
        print("\n🤖 Assistant thinking...")
        
        # The OpenAI call is automatically traced by the integration
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if not function_to_call:
                    print(f"Error: Model tried to call unknown function '{function_name}'")
                    continue

                function_args = json.loads(tool_call.function.arguments)
                
                print(f"🛠️ Calling tool: {function_name}({', '.join(f'{k}={v}' for k, v in function_args.items())})")
                
                # The tool function call is its own span
                function_response = function_to_call(**function_args)
                
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    }
                )
            
            print("🤔 Assistant thinking again after tool use...")
            second_response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
            )
            final_content = second_response.choices[0].message.content
            current_span.set_io(output_data=final_content)
            return final_content
        else:
            final_content = response_message.content
            current_span.set_io(output_data=final_content)
            return final_content

def simulate_conversation(session_id: str, conversation_script: list):
    """Simulates a full conversation, which is a single session."""
    print("\n" + "="*50)
    print(f"🎬 Starting Conversation - Session ID: {session_id}")
    print("="*50 + "\n")
    
    messages = [{"role": "system", "content": "You are a helpful assistant that can use tools."}]
    
    for turn in conversation_script:
        role = turn["role"]
        content = turn["content"]
        
        print(f"👤 User: {content}")
        messages.append({"role": role, "content": content})
        
        # Each turn is a trace within the session.
        # We pass the session_id to the span context manager inside the function.
        # This will create a new trace for this interaction within the session.
        final_response = run_conversation_turn(session_id=session_id, messages=messages)
        
        print(f"\n✨ Assistant: {final_response}")
        messages.append({"role": "assistant", "content": final_response})
        time.sleep(1) # Pause for demo purposes


def main():
    print("🚀 Starting AI Agent Tracing Demo")
    print("This will simulate two conversations with an AI agent, creating two sessions.")
    
    # --- Conversation 1: Using the numpy tool ---
    conversation_1_script = [
        {"role": "user", "content": "Hello, can you help me with a calculation?"},
        {"role": "user", "content": "What is the dot product of the vectors [1, 2, 3] and [4, 5, 6]?"}
    ]
    simulate_conversation(f"agent-conv-{uuid.uuid4().hex[:8]}", conversation_1_script)

    # --- Conversation 2: Using the weather tool ---
    conversation_2_script = [
        {"role": "user", "content": "Hi there! I need a weather update."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    simulate_conversation(f"agent-conv-{uuid.uuid4().hex[:8]}", conversation_2_script)

    print("\n" + "="*50)
    print("✅ Demo completed!")
    print("\n🔍 Check your live monitoring dashboard to see:")
    print("  • 2 Sessions, one for each conversation.")
    print("  • Multiple traces within each session for each user interaction.")
    print("  • Spans for OpenAI calls and tool function executions.")
    print("  • Detailed attributes like token usage, model, and tool parameters.")


if __name__ == "__main__":
    main()
    
    # Ensure all spans are flushed to the backend
    print("\n📤 Flushing all remaining spans...")
    tracer.flush()
    print("✅ All tracing data sent!")

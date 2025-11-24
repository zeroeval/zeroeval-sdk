"""
Example demonstrating automatic prompt optimization with ze.prompt()

This example shows how ZeroEval automatically uses optimized prompt versions
from your dashboard while keeping a fallback in your code.
"""

import zeroeval as ze
from openai import OpenAI

# Initialize ZeroEval
ze.init()
client = OpenAI()

def example_1_auto_optimization():
    """
    When you provide content to ze.prompt(), ZeroEval automatically:
    1. Checks for an optimized version in your dashboard
    2. Uses the optimized version if one exists
    3. Falls back to your provided content if no optimization exists yet
    
    This means your prompts improve automatically without code changes!
    """
    print("=== Example 1: Auto-optimization ===\n")
    
    # This will use the latest optimized version if available in your dashboard
    # Otherwise, it uses the content you provide here
    system_prompt = ze.prompt(
        name="customer-support",
        content="You are a helpful customer support agent."
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I reset my password?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}\n")


def example_2_explicit_content():
    """
    Use from_="explicit" to always use the hardcoded content, bypassing
    auto-optimization. Useful for testing, debugging, or A/B tests.
    """
    print("=== Example 2: Explicit content (bypass optimization) ===\n")
    
    # This ALWAYS uses the hardcoded content, ignoring any optimized versions
    system_prompt = ze.prompt(
        name="customer-support",
        from_="explicit",
        content="You are a helpful customer support agent."
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I reset my password?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}\n")


def example_3_explicit_latest():
    """
    If you want to explicitly require an optimized version to exist,
    use from_="latest". This will fail if no optimized versions exist yet.
    """
    print("=== Example 3: Explicit latest (requires optimization) ===\n")
    
    try:
        # This REQUIRES an optimized version to exist
        system_prompt = ze.prompt(
            name="customer-support",
            from_="latest"
        )
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "How do I reset my password?"}
            ]
        )
        
        print(f"Response: {response.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error: {e}")
        print("This means no optimized versions exist yet. Use content= for fallback.\n")


def example_4_with_variables():
    """
    Variables work seamlessly with all modes.
    """
    print("=== Example 4: Variables with auto-optimization ===\n")
    
    system_prompt = ze.prompt(
        name="company-support",
        content="You are a customer support agent for {{company}}. Be helpful and professional.",
        variables={"company": "TechCorp"}
    )
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "I need help with billing"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}\n")


def example_5_error_handling():
    """
    Demonstrate error handling for invalid usage.
    """
    print("=== Example 5: Error handling ===\n")
    
    try:
        # This will fail: from_="explicit" requires content
        system_prompt = ze.prompt(
            name="customer-support",
            from_="explicit"
        )
    except ValueError as e:
        print(f"✓ Expected error caught: {e}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ZeroEval Auto-Optimization Example")
    print("="*60 + "\n")
    
    example_1_auto_optimization()
    example_2_explicit_content()
    example_3_explicit_latest()
    example_4_with_variables()
    example_5_error_handling()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Use content= for automatic optimization with fallback (RECOMMENDED)")
    print("- Use from_='explicit' to always use hardcoded content")
    print("- Use from_='latest' to require an optimized version")
    print("- Variables work seamlessly with all approaches")
    print("="*60 + "\n")


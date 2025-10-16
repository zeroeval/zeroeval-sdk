#!/usr/bin/env python3
"""
OpenAI A/B Testing Example
==========================

This example shows how to use ze.choose() to A/B test between different
OpenAI models while automatically tracking the results.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing zeroeval
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

import openai
import zeroeval as ze

def main():

    # Initialize ZeroEval (automatically picks up environment variables)
    ze.init()

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create a span for our A/B test
    with ze.span("model_ab_test", tags={"feature": "model_comparison"}):

        # Use ze.choose() to select between two models
        # 70% chance for gpt-4o-mini (faster/cheaper), 30% for gpt-4o (more capable)
        selected_model = ze.choose(
            "model_selection",
            variants={
                "mini": "gpt-4o-mini",
                "full": "gpt-4o"
            },
            weights={
                "mini": 0.7,  # 70% chance
                "full": 0.3   # 30% chance
            }
        )

        print(f"🤖 Selected model: {selected_model}")

        # Make the API call with the selected model
        with ze.span("llm_call", tags={"selected_model": selected_model}):
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains things concisely."},
                    {"role": "user", "content": "Explain the concept of A/B testing in one paragraph."}
                ],
                temperature=0.1,
                max_tokens=150
            )

        # Display the results
        answer = response.choices[0].message.content
        print(answer)

if __name__ == "__main__":
    main()
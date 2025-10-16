#!/usr/bin/env python3
"""
Simple OpenAI Tracing Example
============================

This example shows how to trace OpenAI API calls with ZeroEval.
The tracing happens automatically once you initialize the SDK.
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
    # Initialize ZeroEval (explicitly passing values to ensure they're used)
    ze.init(
    )

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Make a simple OpenAI call - this will be automatically traced
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=100
    )

    print("Response:", response.choices[0].message.content)
    print("✅ OpenAI call completed and automatically traced!")

if __name__ == "__main__":
    main()
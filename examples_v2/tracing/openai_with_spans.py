#!/usr/bin/env python3
"""
OpenAI Tracing with Manual Spans
===============================

This example shows how to add custom spans around OpenAI calls
for better observability and context.
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
        api_key=os.getenv("ZEROEVAL_API_KEY"),
        api_url=os.getenv("ZEROEVAL_API_URL", "http://localhost:8000")
    )

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create a span for the entire conversation
    with ze.span("user_question_answering", tags={"feature": "qa_system"}):

        # Add a span for preprocessing
        with ze.span("question_preprocessing"):
            question = "What is the capital of France?"
            processed_question = question.strip().lower()
            print(f"Original question: {question}")
            print(f"Processed question: {processed_question}")

        # Add a span for the LLM call (this will have nested OpenAI spans automatically)
        with ze.span("llm_generation", tags={"model": "gpt-3.5-turbo"}):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful geography assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=50
            )

        # Add a span for post-processing
        with ze.span("response_postprocessing"):
            answer = response.choices[0].message.content
            formatted_answer = f"Answer: {answer}"
            print(formatted_answer)

    print("✅ Complete conversation traced with custom spans!")

if __name__ == "__main__":
    main()
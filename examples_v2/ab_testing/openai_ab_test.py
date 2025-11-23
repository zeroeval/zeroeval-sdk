#!/usr/bin/env python3
"""
OpenAI A/B Testing Example
==========================

This example shows how to use ze.choose() to A/B test between different
OpenAI models with timeboxed experiments and signal tracking.

Features:
- Timeboxed experiment with duration_days
- Model selection with weighted distribution
- Signal tracking for response quality
- Automatic choice recording
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

    # IMPORTANT: ze.choose() must be called within a span context
    # Signals attached to this span will be linked to the AB test variant
    with ze.span("model_ab_test", tags={"feature": "model_comparison"}) as test_span:

        # Use ze.choose() to select between two models
        # This attaches the ab_choice_id to test_span for signal linkage
        # 70% chance for gpt-4o-mini (faster/cheaper), 30% for gpt-4o (more capable)
        # The experiment runs for 7 days and automatically stops accepting new choices
        selected_model = ze.choose(
            name="model_selection",
            variants={
                "mini": "gpt-4o-mini",
                "full": "gpt-4o"
            },
            weights={
                "mini": 0.7,  # 70% chance
                "full": 0.3   # 30% chance
            },
            duration_days=7,  # Run experiment for 1 week
            default_variant="mini"  # Use mini as fallback after experiment ends
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
        print(f"\n📝 Response:\n{answer}\n")
        
        # Track response quality with signals on the test_span
        # CRITICAL: Attach signals to the SAME SPAN where ze.choose() was called
        # The backend will automatically link these signals to the AB test variant
        # via the span_ab_choices junction table for aggregated analytics
        response_length = len(answer)
        is_concise = response_length <= 200  # Good responses should be concise
        has_good_length = 100 <= response_length <= 200
        
        ze.set_signal(test_span, {
            "response_quality": is_concise,
            "appropriate_length": has_good_length,
            "highly_effective": is_concise and has_good_length  # Both conditions met
        })
        
        print(f"\n🎯 Signals Tracked:")
        print(f"   ✓ response_quality: {is_concise} (≤200 chars)")
        print(f"   ✓ appropriate_length: {has_good_length} (100-200 chars)")
        print(f"   ✓ highly_effective: {is_concise and has_good_length}")
        print(f"\n   Length: {response_length} chars")
        print(f"   💡 These signals are automatically linked to the AB test variant!")
        print(f"\n📊 View the Signal Distribution chart in the ZeroEval dashboard!")
        print(f"   Dashboard path: Monitoring → A/B Testing → model_selection")

if __name__ == "__main__":
    main()
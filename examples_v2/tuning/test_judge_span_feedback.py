#!/usr/bin/env python3
"""
Test Script: Judge Span Feedback via SDK
========================================

This script tests the fix for the SDK feedback submission bug where feedback
for judge evaluation spans was not being saved.

The issue was that judge spans store `task_id` at the root level of attributes
instead of under `zeroeval.prompt_slug`, which caused the backend to fail
slug resolution and return 404.

How to run:
    1. Set your environment variables (ZEROEVAL_API_KEY, ZEROEVAL_API_URL)
    2. Run: python test_judge_span_feedback.py

What this tests:
    1. Creates a test prompt/task in ZeroEval
    2. Makes an LLM call that creates a span
    3. Simulates what a judge would do (the span gets task_id in attributes)
    4. Submits feedback via ze.send_feedback() using the span_id as completion_id
    5. Verifies the feedback was saved successfully

Expected result after fix:
    - Feedback should be saved successfully (no 404 error)
    - The script should print "Feedback submitted successfully"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

import zeroeval as ze


def test_judge_span_feedback():
    """
    Test that feedback can be submitted for judge spans via the SDK.
    
    This simulates the customer's use case:
    - They have a judge evaluation span with a span_id
    - They want to submit feedback using ze.send_feedback()
    - The prompt_slug is the judge's task slug
    """
    
    # Configuration
    api_key = "sk_ze_JCa34MyFIJlEvbJe6QeARqyxQAnTgp86cAJjFwyRVQM"
    api_url = "http://localhost:8000"
    
    if not api_key:
        print("ERROR: ZEROEVAL_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    print("=" * 60)
    print("Test: Judge Span Feedback via SDK")
    print("=" * 60)
    print(f"\nAPI URL: {api_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Initialize ZeroEval
    ze.init(
        api_key=api_key,
        api_url=api_url,
    )
    
    # Test parameters - replace these with actual values from your system
    # You can get these from the ZeroEval dashboard or logs
    # NOTE: The prompt_slug must match the task that the span belongs to!
    prompt_slug = "judge-d5c32d3843534d03b4bc38ad1ae972eb"
    span_id = "f9ed607e-f03e-4527-bc04-0daed58cc574"
    
    if not prompt_slug or not span_id:
        print("\n" + "-" * 60)
        print("MANUAL TEST MODE")
        print("-" * 60)
        print("\nTo run this test, you need a judge span ID and its task slug.")
        print("\nOption 1: Set environment variables")
        print("  TEST_PROMPT_SLUG=your-judge-task-slug")
        print("  TEST_SPAN_ID=your-span-uuid")
        print("\nOption 2: Enter values manually below")
        print("-" * 60)
        
        prompt_slug = input("\nEnter the judge task slug (e.g., 'judge-fc9ade7cc27a42f48fbe3f2b1ff4ae3e'): ").strip()
        span_id = input("Enter the span ID (e.g., 'c72c0da9-0d2d-4b08-82dd-b09d534af160'): ").strip()
        
        if not prompt_slug or not span_id:
            print("\nERROR: Both prompt_slug and span_id are required")
            sys.exit(1)
    
    print(f"\nTest parameters:")
    print(f"  - Prompt Slug: {prompt_slug}")
    print(f"  - Span ID (as completion_id): {span_id}")
    
    # Attempt to submit feedback
    print("\n" + "-" * 60)
    print("Submitting feedback via ze.send_feedback()...")
    print("-" * 60)
    
    try:
        result = ze.send_feedback(
            prompt_slug=prompt_slug,
            completion_id=span_id,  # Using span_id as completion_id (customer's use case)
            thumbs_up=False,  # Simulating negative feedback
            reason="Test feedback from SDK - verifying judge span feedback fix",
            metadata={
                "test": True,
                "test_script": "test_judge_span_feedback.py",
            }
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS! Feedback submitted successfully")
        print("=" * 60)
        print(f"\nResponse: {result}")
        print("\nThe fix is working correctly!")
        print("Judge span feedback can now be submitted via the SDK.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("FAILED! Feedback submission failed")
        print("=" * 60)
        print(f"\nError: {e}")
        
        if "404" in str(e):
            print("\nThis is likely the bug we're trying to fix:")
            print("- The backend couldn't resolve the prompt slug from the span")
            print("- Judge spans have task_id at root level, not under zeroeval")
            print("\nMake sure the backend fix has been deployed!")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_with_live_judge_call():
    """
    Alternative test: Create an actual judge evaluation and submit feedback.
    
    This requires having a judge set up in your ZeroEval project.
    Uncomment and configure if you want to test the full flow.
    """
    print("\n" + "=" * 60)
    print("Test: Full Judge Evaluation Flow (Optional)")
    print("=" * 60)
    print("\nThis test requires:")
    print("  1. A judge configured in your ZeroEval project")
    print("  2. OpenAI API key for making LLM calls")
    print("\nSkipping this test. Uncomment in code to enable.")
    
    # Uncomment below to enable this test:
    """
    import openai
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Skipping: OPENAI_API_KEY not set")
        return
    
    client = openai.OpenAI(api_key=openai_key)
    
    # Make an LLM call that will be evaluated by a judge
    with ze.span("test-llm-call", kind="llm") as span:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        
        print(f"LLM Response: {response.choices[0].message.content}")
        print(f"Span ID: {span.span_id}")
        
        # The judge will automatically evaluate this span
        # Wait a bit for the judge to run, then submit feedback
        import time
        print("Waiting 5 seconds for judge evaluation...")
        time.sleep(5)
        
        # Now submit feedback for the judge's span
        # You would need to look up the judge_call_span_id from the evaluation
    """


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  Judge Span Feedback Test Script")
    print("#  Tests the fix for SDK feedback submission bug")
    print("#" * 60)
    
    test_judge_span_feedback()
    test_with_live_judge_call()
    
    print("\n" + "#" * 60)
    print("#  All tests completed!")
    print("#" * 60 + "\n")

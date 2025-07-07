#!/usr/bin/env python3
"""
ZeroEval Signals Demo
=====================

This example demonstrates how to use ZeroEval's simple, immediate signal system.
Signals are sent directly to the backend when the `ze.signal()` function is called.

This approach is:
- **Simple**: No buffering, queues, or complex async logic.
- **Immediate**: See feedback in your dashboard right away.
- **Flexible**: Works for any entity, even after its spans have completed.
"""

import asyncio
import os
import time
import random
import uuid
from typing import Dict, Any

import zeroeval as ze

# Initialize ZeroEval
ze.init(debug=True, api_key="sk_ze_diTPEUddB7MHvWGSA_NeZEseRd_1_ID3LOwfch0TxQg", api_url="http://localhost:8000")  # Set debug=True to see detailed tracer logs.


@ze.span(name="signals_demo_main", attributes={"demo_version": "2.0"})
async def main():
    """Run a simple demo for the immediate signal sender."""
    display_config_info()
    
    try:
        # Create a span and get its details
        with ze.span(name="user_login_flow", tags={"user_tier": "premium"}):
            await asyncio.sleep(0.2)
            current_span = ze.get_current_span()
            current_trace = ze.get_current_trace()
            current_session = ze.get_current_session()
            print(f"Created base entities (span, trace, session)")

        # --- Simulate signals sent after the fact ---
        
        # 1. User rates their experience (attached to the completed span)
        print("\n--- Simulating User Feedback ---")
        ze.set_signal(current_span, {"user_rating": 5, "feedback_comment": "so_fast"})
        await asyncio.sleep(1)

        # 2. A business event occurs (attached to the session)
        print("\n--- Simulating Business Event ---")
        ze.set_signal(current_session, {"conversion_event": True, "cart_value": 99.99})
        await asyncio.sleep(1)

        # 3. An offline evaluation job completes (attached to the trace)
        print("\n--- Simulating Offline Evaluation ---")
        ze.set_signal(current_trace, {"factual_accuracy": 0.98, "is_safe": True})
        
        print("\n=== Demo Complete ===")
        print("Check the ZeroEval dashboard to see signals in the monitoring UI")

    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure tracer flushes any remaining spans
        from zeroeval.observability.tracer import tracer
        tracer.shutdown()


def display_config_info():
    """Display configuration information."""
    print("=== ZeroEval Immediate Signals Demo ===")
    print("This demo shows the simple, direct signal sending system.\n")
    print(f"API URL: {os.getenv('ZEROEVAL_API_URL', 'Not set')}")
    print(f"Workspace ID: {os.getenv('ZEROEVAL_WORKSPACE_ID', 'Not set')}")
    print(f"API Key: {'Set' if os.getenv('ZEROEVAL_API_KEY') else 'Not set'}")
    
    if not all([os.getenv('ZEROEVAL_API_KEY'), os.getenv('ZEROEVAL_WORKSPACE_ID')]):
        print("\nWarning: Credentials not set. Signals will not be sent.")
        print("   Please set ZEROEVAL_API_KEY and ZEROEVAL_WORKSPACE_ID.\n")


if __name__ == "__main__":
    if not os.getenv('ZEROEVAL_API_URL'):
        os.environ['ZEROEVAL_API_URL'] = 'http://localhost:8000'
    if not os.getenv('ZEROEVAL_WORKSPACE_ID'):
        # Use a random UUID as a placeholder if not set
        os.environ['ZEROEVAL_WORKSPACE_ID'] = str(uuid.uuid4())
        print(f"Using random demo workspace ID: {os.getenv('ZEROEVAL_WORKSPACE_ID')}")
    if not os.getenv('ZEROEVAL_API_KEY'):
        # Use a placeholder for local demo if not set
        os.environ['ZEROEVAL_API_KEY'] = 'sk_ze_placeholder_for_demo'
        print("Using placeholder API key.")
    
    print("\nTo run this demo successfully, please set your real credentials:")
    print("  export ZEROEVAL_API_KEY='sk_ze_...'")
    print("  export ZEROEVAL_WORKSPACE_ID='your-workspace-uuid'")
    print("-" * 60)
    
    asyncio.run(main()) 
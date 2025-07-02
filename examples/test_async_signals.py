#!/usr/bin/env python3
"""
Simple test for async signals functionality
"""

import asyncio
import os
import sys
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

import zeroeval as ze

# Set up test environment
os.environ['ZEROEVAL_API_URL'] = 'http://localhost:8000'
os.environ['ZEROEVAL_WORKSPACE_ID'] = 'test-workspace-123'
os.environ['ZEROEVAL_API_KEY'] = 'test-api-key'

# Initialize ZeroEval
ze.init(api_key="test-api-key")

async def test_async_signals():
    """Simple test for async signals."""
    print("🧪 Testing async signals...")
    
    # Configure async signals
    from zeroeval.observability.tracer import tracer
    tracer.configure(
        async_signals=True,
        signal_api_url="http://localhost:8000",
        signal_api_key="test-api-key"
    )
    
    # Create a simple span and send async signals
    with ze.span(name="test_span", attributes={"test": True}):
        current_span = ze.get_current_span()
        current_trace = ze.get_current_trace()
        current_session = ze.get_current_session()
        
        print(f"📍 Current span: {current_span.span_id}")
        print(f"📍 Current trace: {current_trace}")
        print(f"📍 Current session: {current_session}")
        
        # Send signals to span, trace, and session
        ze.set_signal(current_span, {"test_span_signal": True})
        ze.set_signal(current_trace, {"test_trace_signal": "hello"})
        ze.set_signal(current_session, {"test_session_signal": 42})
        
        print("✅ Signals sent, waiting for async transmission...")
        await asyncio.sleep(3.0)  # Give async writer time to process
    
    print("🔄 Flushing tracer...")
    tracer.flush()
    
    print("⏳ Waiting for final async signal processing...")
    await asyncio.sleep(2.0)
    
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_async_signals()) 
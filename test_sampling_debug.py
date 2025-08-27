#!/usr/bin/env python3
"""Debug script to understand sampling issue."""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_single_trace():
    """Test a single trace to debug the issue."""
    
    # Set environment variable before importing
    os.environ["ZEROEVAL_SAMPLING_RATE"] = "1.0"
    
    # Import fresh
    import zeroeval
    from zeroeval.observability.tracer import Tracer
    
    # Reset singleton
    Tracer._instance = None
    
    # Initialize
    zeroeval.init(api_key="test_key", debug=True)
    
    print(f"Tracer sampling rate: {zeroeval.tracer._sampling_rate}")
    print(f"Tracer instance: {zeroeval.tracer}")
    
    # Create a single span/trace
    span = zeroeval.tracer.start_span("test_trace", is_new_trace=True)
    trace_id = span.trace_id
    
    print(f"Trace ID: {trace_id}")
    print(f"Traces registry: {zeroeval.tracer._traces}")
    
    if trace_id in zeroeval.tracer._traces:
        trace_info = zeroeval.tracer._traces[trace_id]
        print(f"Trace is_sampled: {trace_info.is_sampled}")
        print(f"Trace ref_count: {trace_info.ref_count}")
    else:
        print("ERROR: Trace not in registry!")
    
    # End the span
    zeroeval.tracer.end_span(span)
    
    print(f"After ending span, traces registry: {zeroeval.tracer._traces}")
    print(f"Buffered spans: {len(zeroeval.tracer._spans)}")


if __name__ == "__main__":
    test_single_trace()

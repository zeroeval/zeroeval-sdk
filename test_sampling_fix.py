#!/usr/bin/env python3
"""Test script to verify sampling functionality works correctly."""

import os
import sys
import random
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set seed for reproducible tests
random.seed(42)

def test_sampling_rate(rate, num_traces=1000):
    """Test sampling at a specific rate."""
    print(f"\n=== Testing Sampling Rate: {rate*100}% ({num_traces} traces) ===")
    
    # Set environment variable before importing
    os.environ["ZEROEVAL_SAMPLING_RATE"] = str(rate)
    
    # Force reimport to get fresh tracer instance
    import importlib
    import zeroeval
    from zeroeval.observability.tracer import Tracer
    
    # Reset singleton
    Tracer._instance = None
    
    # Reimport and initialize
    importlib.reload(zeroeval)
    zeroeval.init(api_key="test_key", debug=False)
    
    # Track sampled traces
    sampled_count = 0
    total_count = num_traces
    
    for i in range(total_count):
        # Create a new trace each time
        span = zeroeval.tracer.start_span(f"test_trace_{i}", is_new_trace=True)
        trace_id = span.trace_id
        
        # Check if this trace is sampled
        if trace_id in zeroeval.tracer._traces:
            if zeroeval.tracer._traces[trace_id].is_sampled:
                sampled_count += 1
        
        # End the span properly to clean up
        zeroeval.tracer.end_span(span)
    
    # Calculate actual sampling rate
    actual_rate = sampled_count / total_count
    expected_rate = rate
    
    # Allow for some statistical variance (±5% absolute difference)
    tolerance = 0.05
    is_within_tolerance = abs(actual_rate - expected_rate) <= tolerance
    
    print(f"  Expected rate: {expected_rate*100:.1f}%")
    print(f"  Actual rate: {actual_rate*100:.1f}% ({sampled_count}/{total_count} traces)")
    print(f"  Within tolerance (±{tolerance*100}%): {'✅ YES' if is_within_tolerance else '❌ NO'}")
    
    # Check for memory leaks - all traces should be cleaned up
    remaining_traces = len(zeroeval.tracer._traces)
    print(f"  Memory check - remaining traces: {remaining_traces} {'✅' if remaining_traces == 0 else '❌ MEMORY LEAK!'}")
    
    return is_within_tolerance, actual_rate, remaining_traces == 0


def test_trace_completeness():
    """Test that all spans in a trace follow the same sampling decision."""
    print("\n=== Testing Trace Completeness ===")
    
    os.environ["ZEROEVAL_SAMPLING_RATE"] = "0.5"  # 50% sampling
    
    import importlib
    import zeroeval
    from zeroeval.observability.tracer import Tracer
    
    # Reset singleton
    Tracer._instance = None
    importlib.reload(zeroeval)
    zeroeval.init(api_key="test_key", debug=False)
    
    # Test 10 traces with multiple spans each
    traces_sampled = []
    
    for i in range(10):
        # Start root span (new trace)
        root = zeroeval.tracer.start_span(f"root_{i}", is_new_trace=True)
        trace_id = root.trace_id
        is_sampled = zeroeval.tracer._traces[trace_id].is_sampled
        traces_sampled.append(is_sampled)
        
        # Create child spans in same trace
        child1 = zeroeval.tracer.start_span(f"child1_{i}")
        child2 = zeroeval.tracer.start_span(f"child2_{i}")
        
        # All spans in trace should have same sampling decision
        assert child1.trace_id == trace_id, "Child should have same trace ID"
        assert child2.trace_id == trace_id, "Child should have same trace ID"
        assert zeroeval.tracer._traces[trace_id].is_sampled == is_sampled, "Sampling decision changed!"
        
        # End spans in reverse order (LIFO)
        zeroeval.tracer.end_span(child2)
        zeroeval.tracer.end_span(child1)
        zeroeval.tracer.end_span(root)
        
        # After ending all spans, trace should be cleaned up
        assert trace_id not in zeroeval.tracer._traces, f"Trace {trace_id} not cleaned up!"
    
    sampled = sum(traces_sampled)
    print(f"  Traces sampled: {sampled}/10")
    print(f"  All spans in each trace had consistent sampling: ✅")
    print(f"  All traces cleaned up after completion: ✅")
    

def test_nested_spans_cleanup():
    """Test that nested spans are properly cleaned up even when unsampled."""
    print("\n=== Testing Nested Spans Cleanup ===")
    
    os.environ["ZEROEVAL_SAMPLING_RATE"] = "0"  # Sample nothing
    
    import importlib
    import zeroeval
    from zeroeval.observability.tracer import Tracer
    
    # Reset singleton
    Tracer._instance = None
    importlib.reload(zeroeval)
    zeroeval.init(api_key="test_key", debug=False)
    
    # Create deeply nested spans
    spans = []
    for i in range(5):
        span = zeroeval.tracer.start_span(f"level_{i}", is_new_trace=(i == 0))
        spans.append(span)
    
    trace_id = spans[0].trace_id
    
    # Verify trace is not sampled
    assert not zeroeval.tracer._traces[trace_id].is_sampled, "Trace should not be sampled with rate=0"
    
    # Check active spans stack has all 5 spans
    stack = zeroeval.tracer._active_spans_ctx.get()
    assert len(stack) == 5, f"Expected 5 spans in stack, got {len(stack)}"
    
    # End all spans in reverse order
    for span in reversed(spans):
        zeroeval.tracer.end_span(span)
    
    # Verify stack is empty
    stack = zeroeval.tracer._active_spans_ctx.get()
    assert len(stack) == 0, f"Stack should be empty, but has {len(stack)} spans"
    
    # Verify trace is cleaned up
    assert trace_id not in zeroeval.tracer._traces, "Unsampled trace not cleaned up"
    assert len(zeroeval.tracer._traces) == 0, f"Memory leak: {len(zeroeval.tracer._traces)} traces remain"
    assert len(zeroeval.tracer._spans) == 0, f"No spans should be buffered for unsampled traces, but found {len(zeroeval.tracer._spans)}"
    
    print(f"  Unsampled nested spans: Created 5 levels")
    print(f"  Stack properly cleaned: ✅")
    print(f"  Trace properly cleaned: ✅")
    print(f"  No spans buffered: ✅")


def main():
    print("Testing ZeroEval Sampling Functionality")
    print("="*50)
    
    all_passed = True
    
    # Test different sampling rates
    rates_to_test = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for rate in rates_to_test:
        passed, actual, no_leak = test_sampling_rate(rate, num_traces=500)
        all_passed = all_passed and passed and no_leak
    
    # Test trace completeness
    test_trace_completeness()
    
    # Test cleanup of unsampled spans
    test_nested_spans_cleanup()
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All sampling tests PASSED!")
    else:
        print("❌ Some sampling tests FAILED - check the output above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

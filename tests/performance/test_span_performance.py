"""Performance tests for span operations."""

import gc
import threading
import time

import pytest

from zeroeval.observability import span
from zeroeval.observability.tracer import Tracer


@pytest.mark.performance
def test_many_spans_cpu_performance(tracer: Tracer):
    """Test CPU performance with many spans."""
    num_spans = 1000

    start_time = time.time()

    for i in range(num_spans):
        with span(name=f"span_{i}"):
            pass

    tracer.flush()
    duration = time.time() - start_time

    # Should create 1000 spans in under 5 seconds
    assert duration < 5.0, f"Too slow: {duration:.2f}s for {num_spans} spans"
    assert len(tracer._writer.spans) == num_spans

    # Should achieve at least 200 spans/second
    spans_per_second = num_spans / duration
    assert spans_per_second > 200, f"Too slow: {spans_per_second:.1f} spans/sec"


@pytest.mark.performance
def test_memory_leak_detection(tracer: Tracer):
    """Test for memory leaks in span operations."""

    def create_spans():
        for i in range(100):
            with span(name=f"leak_test_{i}"):
                pass
        tracer.flush()

    # Run multiple batches and check for memory leaks
    gc.collect()
    initial_objects = len(gc.get_objects())

    # Create 5 batches of 100 spans each
    for _ in range(5):
        create_spans()
        gc.collect()

    final_objects = len(gc.get_objects())
    object_growth = final_objects - initial_objects

    # Adjusted threshold - allow for more growth due to test infrastructure and Python version differences
    assert object_growth < 2100, f"Memory leak detected: {object_growth} objects"
    assert len(tracer._writer.spans) == 500


@pytest.mark.performance
def test_concurrent_spans_performance(tracer: Tracer):
    """Test performance with concurrent spans."""
    num_threads = 5
    spans_per_thread = 50

    def create_spans(thread_id):
        for i in range(spans_per_thread):
            with span(name=f"thread_{thread_id}_span_{i}"):
                pass

    start_time = time.time()

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=create_spans, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    tracer.flush()
    duration = time.time() - start_time

    total_spans = num_threads * spans_per_thread
    assert duration < 10.0, f"Concurrent spans too slow: {duration:.2f}s"
    assert len(tracer._writer.spans) == total_spans


@pytest.mark.performance
def test_span_creation_speed(tracer: Tracer):
    """Test that spans can be created at reasonable speed."""
    iterations = 1000

    # Test span creation speed
    @span(name="speed_test")
    def traced_func():
        return 42

    start = time.time()
    for _ in range(iterations):
        traced_func()
    duration = time.time() - start

    tracer.flush()

    # Should create spans at reasonable speed
    spans_per_second = iterations / duration
    assert spans_per_second > 500, (
        f"Span creation too slow: {spans_per_second:.1f} spans/sec"
    )
    assert duration < 5.0, (
        f"Span creation took too long: {duration:.2f}s for {iterations} spans"
    )
    assert len(tracer._writer.spans) == iterations


@pytest.mark.performance
def test_buffer_efficiency(tracer: Tracer):
    """Test buffer management efficiency."""
    original_max = tracer._max_spans
    tracer._max_spans = 50  # Small buffer for testing

    try:
        # Create more spans than buffer size
        for i in range(200):
            with span(name=f"buffer_test_{i}"):
                pass

        tracer.flush()

        # Should have all spans despite small buffer
        assert len(tracer._writer.spans) == 200

    finally:
        tracer._max_spans = original_max


@pytest.mark.performance
def test_deep_nesting_performance(tracer: Tracer):
    """Test performance with deep nesting."""
    depth = 100

    def create_nested(current_depth):
        if current_depth <= 0:
            return

        with span(name=f"nested_{current_depth}"):
            create_nested(current_depth - 1)

    start_time = time.time()
    create_nested(depth)
    duration = time.time() - start_time

    tracer.flush()

    # Should handle deep nesting efficiently
    assert duration < 2.0, f"Deep nesting too slow: {duration:.2f}s"
    assert len(tracer._writer.spans) == depth

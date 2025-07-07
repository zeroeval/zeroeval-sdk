import pytest

from zeroeval.observability import span
from zeroeval.observability.tracer import Tracer


@pytest.mark.core
def test_create_simple_trace(tracer: Tracer):
    """Tests that a simple parent-child trace is created and flushed correctly."""
    # Act
    with span(name="parent"):
        with span(name="child"):
            pass

    tracer.flush()

    # Assert
    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 2

    parent = next(s for s in mock_writer.spans if s["name"] == "parent")
    child = next(s for s in mock_writer.spans if s["name"] == "child")

    assert parent["parent_id"] is None
    assert child["parent_id"] == parent["span_id"]
    assert child["trace_id"] == parent["trace_id"]


@pytest.mark.core
def test_tracer_is_thread_safe(tracer: Tracer):
    """Tests that the tracer handles spans from multiple threads correctly."""
    import threading

    def create_trace(name: str):
        with span(name=f"parent-{name}"):
            with span(name=f"child-{name}"):
                pass

    threads = []
    for i in range(5):
        thread = threading.Thread(target=create_trace, args=(f"thread-{i}",))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    tracer.flush()

    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 10  # 5 threads * 2 spans

    # Check that each trace is consistent
    for i in range(5):
        parent = next(s for s in mock_writer.spans if s["name"] == f"parent-thread-{i}")
        child = next(s for s in mock_writer.spans if s["name"] == f"child-thread-{i}")
        assert child["parent_id"] == parent["span_id"]
        assert child["trace_id"] == parent["trace_id"]


@pytest.mark.core
def test_tracer_shutdown(tracer: Tracer):
    """Tests that the tracer stops accepting spans after shutdown."""
    with span(name="span_before_shutdown"):
        pass

    tracer.shutdown()

    # This span should be ignored and return a no-op span
    with span(name="span_after_shutdown") as s:
        assert s.name == "noop_span"

    # The flush in shutdown should have sent the first span.
    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 1
    assert mock_writer.spans[0]["name"] == "span_before_shutdown"


@pytest.mark.core
def test_auto_flush_on_max_spans(tracer: Tracer):
    """Tests that the buffer is flushed automatically when it reaches max capacity."""
    tracer._max_spans = 5  # Set a low limit for testing

    for i in range(5):
        # Each trace is 1 span, so it shouldn't trigger the flush until the 5th one.
        with span(name=f"span-{i}"):
            pass

    # The 5th span should trigger a flush
    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 5

    # Another span should not be in the buffer yet
    with span(name="one_more"):
        pass

    assert len(mock_writer.spans) == 5
    tracer.flush()
    assert len(mock_writer.spans) == 6

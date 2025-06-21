import pytest
from typing import List, Dict, Any

from zeroeval.observability.tracer import Tracer
from zeroeval.observability.writer import SpanWriter


class MockSpanWriter(SpanWriter):
    """A mock writer that stores spans in a list for inspection."""

    def __init__(self):
        self.spans = []
        self.cleared = False

    def write(self, spans: List[Dict[str, Any]]) -> None:
        self.spans.extend(spans)

    def clear(self):
        self.spans.clear()


@pytest.fixture
def tracer():
    """Fixture to get a clean tracer instance with a mock writer for each test."""
    # Since Tracer is a singleton, we need to be careful with its state.
    t = Tracer()
    mock_writer = MockSpanWriter()
    
    # Monkeypatch the writer instance
    original_writer = t._writer
    t._writer = mock_writer
    
    # Clean up any leftover state from other tests
    t._spans.clear()
    t._trace_buckets.clear()
    t._trace_counts.clear()
    
    yield t
    
    # Teardown
    t.flush()
    mock_writer.clear()
    # Restore the original writer to avoid side effects in other tests
    t._writer = original_writer
    # Reset shutdown flag for subsequent tests
    t._shutdown_called = False 
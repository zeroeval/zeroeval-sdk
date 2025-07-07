import sys
from typing import Any, Dict, List

import pytest

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
    t._traces.clear()
    t._active_spans_ctx.set([])

    yield t

    # Teardown
    t.flush()
    mock_writer.clear()
    # Restore the original writer to avoid side effects in other tests
    t._writer = original_writer
    # Reset shutdown flag for subsequent tests
    t._shutdown_called = False


@pytest.fixture
def python_version():
    """Fixture that provides current Python version info."""
    return sys.version_info


@pytest.fixture
def mock_span_writer():
    """Fixture that provides a clean MockSpanWriter instance."""
    return MockSpanWriter()

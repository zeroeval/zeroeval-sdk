"""Global test configuration for ZeroEval SDK tests."""

import sys
from typing import Any, Dict, List

import pytest

from zeroeval.observability.tracer import Tracer
from zeroeval.observability.writer import SpanWriter


class MockSpanWriter(SpanWriter):
    """A mock writer that stores spans for testing."""

    def __init__(self):
        self.spans = []

    def write(self, spans: List[Dict[str, Any]]) -> None:
        self.spans.extend(spans)

    def clear(self):
        self.spans.clear()


@pytest.fixture
def tracer():
    """Fixture for a clean tracer instance."""
    t = Tracer()
    mock_writer = MockSpanWriter()

    # Store original writer
    original_writer = t._writer
    t._writer = mock_writer

    # Clean up state
    t._spans.clear()
    t._traces.clear()
    t._active_spans_ctx.set([])

    yield t

    # Cleanup
    t.flush()
    mock_writer.clear()
    t._writer = original_writer
    t._shutdown_called = False


@pytest.fixture
def python_version():
    """Current Python version as tuple."""
    return sys.version_info[:2]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "core: Core functionality tests")
    config.addinivalue_line("markers", "performance: Performance tests")


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--runperformance",
        action="store_true",
        default=False,
        help="Run performance tests (skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip performance tests unless --runperformance is given."""
    if not config.getoption("--runperformance"):
        skip_perf = pytest.mark.skip(
            reason="Performance tests skipped. Use --runperformance to run."
        )
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_perf)

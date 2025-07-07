import pytest

from zeroeval.observability import span
from zeroeval.observability.tracer import Tracer


@pytest.mark.core
def test_decorator_success(tracer: Tracer):
    """Tests that the @span decorator wraps a function and records a span."""

    @span(name="my_decorated_function")
    def my_func(a, b):
        return a + b

    result = my_func(1, 2)

    tracer.flush()

    assert result == 3
    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 1

    s = mock_writer.spans[0]
    assert s["name"] == "my_decorated_function"
    assert s["status"] == "ok"
    assert '"a": "1"' in s["input_data"]
    assert '"b": "2"' in s["input_data"]
    assert s["output_data"] == "3"


@pytest.mark.core
def test_decorator_exception(tracer: Tracer):
    """Tests that the @span decorator correctly records an exception."""

    @span(name="my_failing_function")
    def my_func():
        raise ValueError("This is a test error")

    with pytest.raises(ValueError, match="This is a test error"):
        my_func()

    tracer.flush()

    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 1

    s = mock_writer.spans[0]
    assert s["name"] == "my_failing_function"
    assert s["status"] == "error"
    assert s["error_code"] == "ValueError"
    assert s["error_message"] == "This is a test error"
    assert "Traceback" in s["error_stack"]


@pytest.mark.core
@pytest.mark.asyncio
async def test_decorator_async_success(tracer: Tracer):
    """Tests that the @span decorator correctly wraps an async function."""

    @span(name="my_async_function")
    async def my_async_func(a, b):
        return a + b

    result = await my_async_func(3, 4)

    tracer.flush()

    assert result == 7
    mock_writer = tracer._writer
    assert len(mock_writer.spans) == 1

    s = mock_writer.spans[0]
    assert s["name"] == "my_async_function"
    assert s["status"] == "ok"
    assert s["output_data"] == "7"

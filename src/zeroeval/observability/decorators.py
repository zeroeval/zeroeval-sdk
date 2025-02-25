import functools
import inspect
from typing import Optional, Dict, Any, Callable, TypeVar, cast

from .tracer import tracer

F = TypeVar('F', bound=Callable[..., Any])


class span:
    """
    Decorator and context manager for creating spans around code blocks.
    
    Usage as decorator:
        @span(name="operation_name")
        def my_function():
            ...
    
    Usage as context manager:
        with span(name="operation_name"):
            ...
    """
    
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self._span = None
    
    def __call__(self, func: F) -> F:
        """Use as a decorator to trace a function."""
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self:
                    return await func(*args, **kwargs)
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self:
                    return func(*args, **kwargs)
            return cast(F, wrapper)
    
    def __enter__(self):
        """Start a span when entering a context."""
        self._span = tracer.start_span(name=self.name, attributes=self.attributes)
        return self._span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span when exiting the context."""
        if exc_type:
            # Add error information to the span
            if self._span:
                self._span.attributes.update({
                    "error": True,
                    "error.type": exc_type.__name__,
                    "error.message": str(exc_val)
                })
        
        if self._span:
            tracer.end_span(self._span)
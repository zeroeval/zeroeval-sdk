import functools
import inspect
import traceback
import json
from typing import Optional, Dict, Any, Callable, TypeVar, cast

from .tracer import tracer

F = TypeVar('F', bound=Callable[..., Any])


class span:
    """
    Decorator and context manager for creating spans around code blocks.
    
    Usage as decorator:
        @span(name="operation_name")
        def my_function(x, y):
            return x + y  # Parameters and return value will be captured
    
    Usage as decorator with manual I/O:
        @span(name="operation_name", input_data="manual input", output_data="manual output")
        def my_function():
            ...
    
    Usage as context manager:
        with span(name="operation_name") as current_span:
            result = do_something()
            current_span.set_io(input_data="my input", output_data=str(result))
    """
    
    def __init__(
        self, 
        name: str, 
        session_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None
    ):
        self.name = name
        self.session_id = session_id
        self.attributes = attributes or {}
        self.manual_input = input_data
        self.manual_output = output_data
        self._span = None

    def _capture_args_as_input(self, args: tuple, kwargs: dict, func: Callable) -> str:
        """Convert function arguments to a JSON string representation."""
        # Get function parameter names
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Create a dictionary of args and kwargs
        args_dict = {}
        
        # Add positional arguments
        for i, arg in enumerate(args):
            if i < len(params):
                args_dict[params[i]] = str(arg)
            else:
                args_dict[f"arg{i}"] = str(arg)
        
        # Add keyword arguments
        for key, value in kwargs.items():
            args_dict[key] = str(value)
            
        return json.dumps(args_dict, indent=2)
    
    def __call__(self, func: F) -> F:
        """Use as a decorator to trace a function."""
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self as current_span:
                    # Capture function source code and context if enabled
                    if tracer.collect_code_details:
                        try:
                            current_span.set_code(inspect.getsource(func))
                            # Get the source file and line number of the decorated function
                            filepath = inspect.getsourcefile(func)
                            lineno = inspect.getsourcelines(func)[1]
                            current_span.set_code_context(filepath=filepath, lineno=lineno)
                        except (OSError, TypeError):
                            pass  # Fail silently if introspection fails
                    
                    # Capture input parameters if no manual input provided
                    if self.manual_input is None:
                        current_span.set_io(
                            input_data=self._capture_args_as_input(args, kwargs, func)
                        )
                    else:
                        current_span.set_io(input_data=self.manual_input)
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Capture output if no manual output provided
                    if self.manual_output is None:
                        current_span.set_io(output_data=str(result))
                    else:
                        current_span.set_io(output_data=self.manual_output)
                    
                    return result
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self as current_span:
                    # Capture function source code and context if enabled
                    if tracer.collect_code_details:
                        try:
                            current_span.set_code(inspect.getsource(func))
                            # Get the source file and line number of the decorated function
                            filepath = inspect.getsourcefile(func)
                            lineno = inspect.getsourcelines(func)[1]
                            current_span.set_code_context(filepath=filepath, lineno=lineno)
                        except (OSError, TypeError):
                            pass  # Fail silently if introspection fails
                    
                    # Capture input parameters if no manual input provided
                    if self.manual_input is None:
                        current_span.set_io(
                            input_data=self._capture_args_as_input(args, kwargs, func)
                        )
                    else:
                        current_span.set_io(input_data=self.manual_input)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Capture output if no manual output provided
                    if self.manual_output is None:
                        current_span.set_io(output_data=str(result))
                    else:
                        current_span.set_io(output_data=self.manual_output)
                    
                    return result
            return cast(F, wrapper)
    
    def __enter__(self):
        """Start a span when entering a context."""
        self._span = tracer.start_span(
            name=self.name, 
            attributes=self.attributes,
            session_id=self.session_id
        )
        
        # If code collection is enabled, capture the calling context.
        if tracer.collect_code_details:
            try:
                # Go up 2 frames to get the caller of the 'with span(...)' statement
                frame = inspect.currentframe().f_back.f_back
                self._span.set_code_context(
                    filepath=frame.f_code.co_filename,
                    lineno=frame.f_lineno
                )
            except:
                pass # Fail silently if introspection fails
        
        if self.manual_input is not None or self.manual_output is not None:
            self._span.set_io(self.manual_input, self.manual_output)
        return self._span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the span when exiting the context."""
        if exc_type and self._span:
            self._span.set_error(
                code=exc_type.__name__,
                message=str(exc_val),
                stack=traceback.format_exc()
            )
        
        if self._span:
            tracer.end_span(self._span)
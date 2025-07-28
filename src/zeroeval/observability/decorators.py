import ast
import functools
import inspect
import json
import logging
import os
import traceback
from typing import Any, Callable, Optional, TypeVar, Union, cast

from .tracer import tracer

F = TypeVar('F', bound=Callable[..., Any])
logger = logging.getLogger(__name__)


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
        kind: str = "generic",
        session_id: Optional[str] = None,
        session: Optional[Union[str, dict[str, str]]] = None,
        attributes: Optional[dict[str, Any]] = None,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None,
        tags: Optional[dict[str, str]] = None
    ):
        self.name = name
        self.kind = kind
        self.attributes = attributes or {}
        self.manual_input = input_data
        self.manual_output = output_data
        self._span = None
        self._tags = tags or {}
        
        # Handle session parameter - support both legacy and new format
        self._session_id = None
        self._session_name = None
        
        if session is not None:
            if isinstance(session, dict):
                # New format: session={"id": "...", "name": "..."}
                self._session_id = session.get("id")
                self._session_name = session.get("name")
            elif isinstance(session, str):
                # Also support session as a string (just ID)
                self._session_id = session
        elif session_id is not None:
            # Legacy format: session_id="..."
            self._session_id = session_id

    def _capture_decorated_function_code(self, func: Callable):
        """Captures the function signature and body, excluding the decorator."""
        try:
            # Get the full source including the decorator
            full_source = inspect.getsource(func)
            
            # The 'def' keyword for a function marks the start of what we want to capture.
            # inspect.getsource() on a decorated function returns the decorator source too.
            # We find the start of the 'def' or 'async def' to strip off the decorator.
            def_keyword = "async def " if inspect.iscoroutinefunction(func) else "def "
            def_start_pos = full_source.find(def_keyword)

            if def_start_pos != -1:
                code = full_source[def_start_pos:]
            else:
                # Fallback to the full source if we can't find the 'def' keyword
                code = full_source
            
            self._span.set_code(code)

            # Set file context
            filepath = inspect.getsourcefile(func)
            lineno = inspect.getsourcelines(func)[1]
            self._span.set_code_context(filepath=filepath, lineno=lineno)

        except (OSError, TypeError):
            logger.debug(f"Failed to inspect function source for '{func.__name__}'.", exc_info=True)

    def _capture_code_from_context(self, frame: inspect.FrameInfo):
        """
        Tries to capture the source code inside the `with` block using AST parsing.
        """
        try:
            filepath = frame.f_code.co_filename
            lineno = frame.f_lineno
            self._span.set_code_context(filepath=filepath, lineno=lineno)

            # Avoid trying to parse non-project files or virtual envs
            if not os.path.exists(filepath) or "site-packages" in filepath or "lib/python" in filepath:
                 return

            with open(filepath, encoding='utf-8') as f:
                source_file_content = f.read()

            tree = ast.parse(source_file_content, filename=filepath)

            class WithVisitor(ast.NodeVisitor):
                def __init__(self, line_number):
                    self.line_number = line_number
                    self.with_node = None

                def visit_With(self, node: ast.With):
                    if node.lineno == self.line_number:
                        self.with_node = node
                    self.generic_visit(node)
            
            visitor = WithVisitor(lineno)
            visitor.visit(tree)

            node = visitor.with_node
            if node and node.body and hasattr(node.body[-1], 'end_lineno'):
                body_start_lineno = node.body[0].lineno
                body_end_lineno = node.body[-1].end_lineno
                
                source_lines = source_file_content.splitlines()
                code_lines = source_lines[body_start_lineno - 1 : body_end_lineno]
                
                if code_lines:
                    # Dedent the code block
                    indentation = len(code_lines[0]) - len(code_lines[0].lstrip(' '))
                    dedented_lines = [line[indentation:] for line in code_lines]
                    code = "\n".join(dedented_lines)
                    self._span.set_code(code)

        except Exception:
            logger.debug("Failed to inspect frame context or parse source code.", exc_info=True)


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
                        self._capture_decorated_function_code(func)
                    
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
                        self._capture_decorated_function_code(func)
                    
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
            kind=self.kind,
            attributes=self.attributes,
            session_id=self._session_id,
            session_name=self._session_name,
            tags=self._tags
        )
        
        if tracer.collect_code_details:
            # Go up 1 frame – the caller is the frame that owns the `with` line
            frame = inspect.currentframe()
            if frame and frame.f_back:
                self._capture_code_from_context(frame.f_back)
        
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
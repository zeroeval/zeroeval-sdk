import functools
import inspect
from typing import Any, Callable, Optional


def task(outputs: list[str]) -> Callable:
    """
    Decorator to mark a function as a task that can be run on a dataset.
    
    Args:
        outputs: List of output column names this task produces
        
    Example:
        @task(outputs=["pred"])
        def solve(row):
            return {"pred": llm_answer(row.question)}
    """
    def decorator(func: Callable) -> Callable:
        # Store metadata on the function
        func._is_task = True
        func._outputs = outputs
        func._task_name = func.__name__
        func._task_code = inspect.getsource(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Validate outputs
            if not isinstance(result, dict):
                raise TypeError(f"Task {func.__name__} must return a dictionary, got {type(result)}")
            
            missing_outputs = set(outputs) - set(result.keys())
            if missing_outputs:
                raise ValueError(f"Task {func.__name__} missing outputs: {missing_outputs}")
                
            return result
            
        # Preserve the metadata on the wrapper
        wrapper._is_task = True
        wrapper._outputs = outputs
        wrapper._task_name = func.__name__
        wrapper._task_code = func._task_code
        
        return wrapper
    
    return decorator 
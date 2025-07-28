import functools
import inspect
from typing import Any, Callable, Optional, Union, List, Dict
from enum import Enum


class EvaluationMode(Enum):
    ROW = "row"
    COLUMN = "column"  
    RUN = "run"


class Evaluation:
    """Represents a registered evaluation function with metadata."""
    
    def __init__(
        self,
        func: Callable,
        mode: EvaluationMode,
        outputs: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.func = func
        self.mode = mode
        self.outputs = outputs
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self._code = inspect.getsource(func)
        
    def __call__(self, *args, **kwargs):
        """Make the evaluation callable."""
        return self.func(*args, **kwargs)
        
    def __repr__(self):
        return f"Evaluation({self.name}, mode={self.mode.value})"


# Global registry for evaluations
_registered_evaluations: Dict[str, Evaluation] = {}


def evaluation(
    mode: str = "row",
    outputs: Optional[List[str]] = None,
    name: Optional[str] = None
) -> Callable:
    """
    Decorator to register an evaluation function.
    
    Args:
        mode: Evaluation mode - "row", "column", or "run"
        outputs: List of output field names this evaluation produces
        name: Optional custom name for the evaluation
        
    Examples:
        # Row evaluation - gets full row access
        @evaluation(mode="row", outputs=["exact_match"])
        def exact_match(row):
            return {"exact_match": int(row["prediction"] == row["answer"])}
            
        # Column evaluation - gets all rows
        @evaluation(mode="column", outputs=["f1_score"])
        def f1_score(dataset):
            predictions = [row["prediction"] for row in dataset]
            labels = [row["label"] for row in dataset]
            return {"f1_score": calculate_f1(predictions, labels)}
            
        # Run evaluation - gets all runs
        @evaluation(mode="run", outputs=["pass_at_3"])
        def pass_at_k(runs, k=3):
            # Evaluate across multiple runs
            return {"pass_at_3": calculate_pass_at_k(runs, k)}
    """
    if outputs is None:
        outputs = []
        
    def decorator(func: Callable) -> Evaluation:
        eval_mode = EvaluationMode(mode)
        
        # Create evaluation wrapper
        eval_obj = Evaluation(
            func=func,
            mode=eval_mode,
            outputs=outputs,
            name=name or func.__name__
        )
        
        # Register in global registry
        _registered_evaluations[eval_obj.name] = eval_obj
        
        # Add metadata to the function
        func._is_evaluation = True
        func._evaluation_mode = eval_mode
        func._outputs = outputs
        
        return eval_obj
    
    return decorator


def get_evaluation(name: str) -> Optional[Evaluation]:
    """Get a registered evaluation by name."""
    return _registered_evaluations.get(name)


def get_all_evaluations() -> Dict[str, Evaluation]:
    """Get all registered evaluations."""
    return _registered_evaluations.copy()


def clear_evaluations():
    """Clear all registered evaluations (useful for testing)."""
    _registered_evaluations.clear() 
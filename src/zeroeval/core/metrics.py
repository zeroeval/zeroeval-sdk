"""
Column and Run Metrics System

Separate from row evaluations to avoid schema conflicts.
Simple, clean decorators for aggregate metrics.
"""

from typing import Callable, Dict, Any, List
from functools import wraps

# Global registries for metrics
_column_metrics = {}
_run_metrics = {}


class ColumnMetric:
    def __init__(self, name: str, func: Callable, outputs: List[str]):
        self.name = name
        self.func = func
        self.outputs = outputs
    
    def __call__(self, dataset):
        return self.func(dataset)


class RunMetric:
    def __init__(self, name: str, func: Callable, outputs: List[str]):
        self.name = name
        self.func = func
        self.outputs = outputs
    
    def __call__(self, runs):
        return self.func(runs)


def column_metric(outputs: List[str]):
    """Decorator for column-level metrics that operate on entire dataset."""
    def decorator(func: Callable):
        metric = ColumnMetric(
            name=func.__name__,
            func=func,
            outputs=outputs
        )
        _column_metrics[func.__name__] = metric
        return metric
    return decorator


def run_metric(outputs: List[str]):
    """Decorator for run-level metrics that operate across multiple runs."""
    def decorator(func: Callable):
        metric = RunMetric(
            name=func.__name__,
            func=func,
            outputs=outputs
        )
        _run_metrics[func.__name__] = metric
        return metric
    return decorator


def get_column_metric(name: str) -> ColumnMetric:
    """Get a column metric by name."""
    return _column_metrics.get(name)


def get_run_metric(name: str) -> RunMetric:
    """Get a run metric by name."""
    return _run_metrics.get(name) 
"""RunCollection class for elegant multi-run operations."""
from typing import List, Union, Callable, Optional, Any


class RunCollection:
    """A collection of runs that provides a fluent interface for batch operations."""
    
    def __init__(self, runs: List["Run"]):
        """Initialize with a list of Run objects.
        
        Args:
            runs: List of Run objects to manage
        """
        if not runs:
            raise ValueError("RunCollection requires at least one run")
        self.runs = runs
        
    def eval(self, evaluators: List[Union[Callable, Any]]) -> "RunCollection":
        """Apply evaluators to all runs in the collection.
        
        Args:
            evaluators: List of evaluators to apply
            
        Returns:
            Self for method chaining
        """
        for run in self.runs:
            run.eval(evaluators)
        return self
        
    def column_metrics(self, metrics: List[Union[Callable, Any]]) -> "RunCollection":
        """Apply column metrics to all runs in the collection.
        
        Args:
            metrics: List of column metrics to apply
            
        Returns:
            Self for method chaining
        """
        for run in self.runs:
            run.column_metrics(metrics)
        return self
        
    def run_metrics(self, metrics: List[Union[Callable, Any]]) -> "RunCollection":
        """Apply run metrics across all runs in the collection.
        
        Args:
            metrics: List of run metrics to apply
            
        Returns:
            Self for method chaining
        """
        # Run metrics need access to all runs, so we pass them to the first run
        if self.runs:
            self.runs[0].run_metrics(metrics, self.runs)
        return self
        
    def __len__(self) -> int:
        """Return the number of runs in the collection."""
        return len(self.runs)
        
    def __getitem__(self, index: int) -> "Run":
        """Get a specific run by index."""
        return self.runs[index]
        
    def __iter__(self):
        """Iterate over runs in the collection."""
        return iter(self.runs)
        
    def __repr__(self) -> str:
        """String representation of the collection."""
        return f"RunCollection({len(self.runs)} runs)"
        
    @property
    def first(self) -> "Run":
        """Get the first run in the collection."""
        return self.runs[0] if self.runs else None
        
    @property
    def last(self) -> "Run":
        """Get the last run in the collection."""
        return self.runs[-1] if self.runs else None
        
    def to_list(self) -> List["Run"]:
        """Convert back to a list of runs if needed."""
        return self.runs

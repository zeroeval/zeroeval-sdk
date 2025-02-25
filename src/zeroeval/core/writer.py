from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any
import json

if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .experiment_class import ExperimentResult

class DatasetWriter(ABC):
    """Interface for writing datasets to different destinations."""
    
    @abstractmethod
    def write(self, dataset: 'Dataset') -> None:
        """Write a dataset to the destination."""
        pass


class DatasetConsoleWriter(DatasetWriter):
    """Writes datasets to the console for debugging."""
    
    def write(self, dataset: 'Dataset') -> None:
        """Print dataset to the console in a readable format."""
        print(f"Dataset: {dataset.name}")
        if dataset.description:
            print(f"Description: {dataset.description}")
        print(f"Records: {len(dataset)}")
        print(f"Columns: {', '.join(dataset.columns)}")
        print("\nSample data:")
        for i, record in enumerate(dataset):
            if i >= 5:  # Show at most 5 records
                print("...")
                break
            print(f"  {i+1}: {record}")


class ExperimentResultWriter(ABC):
    """Interface for writing experiment results to different destinations."""
    
    @abstractmethod
    def write(self, dataset: 'Dataset') -> None:
        """Write experiment results to the destination."""
        pass


class ExperimentResultConsoleWriter(ExperimentResultWriter):
    """Writes experiment results to the console for debugging."""
    
    def write(self, experiment_result: 'ExperimentResult') -> None:
        """Print experiment results to the console in a readable format."""
        print(f"Writing experiment results to console: {experiment_result.result}")

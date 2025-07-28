from typing import Any, Optional

from .writer import EvaluatorBackendWriter, EvaluatorWriter


class Evaluator:
    """
    A class that defines an evaluator for an experiment.
    """
    def __init__(self, name: str, code: str, description: str, experiment_id: str, evaluation_mode: str = "row"):
        self.name = name
        self.code = code
        self.description = description  
        self.experiment_id = experiment_id
        self.evaluation_mode = evaluation_mode
        self._backend_id = None
        # Default to console writer for now, can be changed to backend writer later
        self._writer: EvaluatorWriter = EvaluatorBackendWriter()

    def _write(self) -> Optional[str]:
        """Writes the evaluator to the writer if it hasn't been written yet."""
        if not self._backend_id:
            assigned_id = self._writer._write(self)
            if assigned_id:
                self._backend_id = assigned_id
        return self._backend_id

class Evaluation:
    """
    A class that defines the result of an evaluator.
    """
    def __init__(self, evaluator: Evaluator, result: Any, experiment_result_id: str, dataset_row_id: str):
        self.evaluator = evaluator
        self.result = result
        self.experiment_result_id = experiment_result_id
        self.dataset_row_id = dataset_row_id
        # Use the same writer type as the parent evaluator
        self._writer = evaluator._writer
    
    def _write(self) -> None:
        """Write this Evaluation to the writer."""
        self._writer._write(self)

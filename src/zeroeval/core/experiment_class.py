from typing import List, Callable, Any, Optional
from .dataset_class import Dataset
from .writer import ExperimentResultWriter, ExperimentResultBackendWriter
import inspect

class Experiment:
    """
    Represents an experiment that can run a 'task' (the user's function)
    on each row of a dataset, and then evaluate results with 'evaluators'.
    """

    def __init__(
        self, 
        dataset: Dataset, 
        task: Callable[[Any], Any], 
        evaluators: Optional[List[Callable[[Any, Any], Any]]] = None,
        name: Optional[str] = None,
        code: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.dataset = dataset
        self.task = task
        self.evaluators = evaluators or []

        # If user didn't provide a name, try using the function's __name__
        self.name = name or (task.__name__ if hasattr(task, "__name__") else "unnamed_experiment")

        # If user didn't provide code, fall back to source code of task (if possible)
        if code is not None:
            self.code = code
        else:
            try:
                self.code = inspect.getsource(task)
            except OSError:
                self.code = None

        # If user didn't provide a description, fall back to docstring of task (if any)
        self.description = description or (task.__doc__ or "")

        # We'll treat the default writer as a backend writer.
        self._writer: ExperimentResultWriter = ExperimentResultBackendWriter()
        self.results: List['ExperimentResult'] = []

        # Will be set once the experiment is persisted to the backend
        self._backend_id: Optional[str] = None

    def run_task(self, subset: Optional[List[dict]] = None) -> List['ExperimentResult']:
        """
        Run the task function on each row (either a given subset or the entire dataset).
        Store the output in self.results and automatically write each result to the backend.
        """
        # Write (or ensure writing) of the Experiment to the backend, retrieving an experiment_id
        experiment_id = self._write(self._writer)  # sets self._backend_id internally
        if not experiment_id:
            print("[Experiment] Could not create or retrieve experiment ID from writer.")
            return []

        # If subset is given, we assume it includes 'row_id' if needed
        rows_to_run = subset if subset is not None else self.dataset._get_all_full_rows()
        self.results = []

        for row_data in rows_to_run:
            row_id = row_data.get("row_id") if isinstance(row_data, dict) else None
            # "data" sub-dict or entire row
            row_content = row_data["data"] if isinstance(row_data, dict) and "data" in row_data else row_data

            task_output = self.task(row_content)
            
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                row_data=row_data,
                row_id=row_id,
                result=task_output
            )
            experiment_result._write(self._writer)
            self.results.append(experiment_result)

        return self.results

    def run_evaluators(
        self, 
        evaluators: Optional[List[Callable[[Any, Any], Any]]] = None, 
        results: Optional[List['ExperimentResult']] = None
    ) -> List['ExperimentResult']:
        """
        Run the specified evaluators on a list of results (or on self.results if none provided).
        Each evaluator is a function: evaluator(row_data, result).
        """
        if evaluators is None:
            evaluators = self.evaluators
        if results is None:
            results = self.results

        for experiment_result in results:
            experiment_result.evaluations = {}

            # For convenience in the evaluator, pass the row's "data" portion only
            row_data_for_eval = (
                experiment_result.row_data["data"]
                if isinstance(experiment_result.row_data, dict)
                   and "data" in experiment_result.row_data
                else experiment_result.row_data
            )

            for evaluator in evaluators:
                evaluation_output = evaluator(row_data_for_eval, experiment_result.result)
                # Store evaluator's output under the function's __name__
                experiment_result.evaluations[evaluator.__name__] = evaluation_output

        return results

    def run(self, subset: Optional[List[dict]] = None) -> List['ExperimentResult']:
        """
        Convenience method to run both the task on the dataset
        and then run all evaluators.
        """
        self.run_task(subset=subset)
        self.run_evaluators()
        return self.results

    def _write(self, writer: 'ExperimentResultWriter') -> Optional[str]:
        """Writes the experiment to the writer if it hasn't been written yet."""
        if not self._backend_id:
            assigned_id = writer._write(self)
            if assigned_id:
                self._backend_id = assigned_id
        return self._backend_id


class ExperimentResult:
    """
    Represents the result of running the experiment's task on a single row.
    Includes optional row_id for referencing dataset rows in a backend.
    """
    def __init__(
        self,
        experiment_id: str,
        row_data: Optional[dict],
        row_id: Optional[str],
        result: Any
    ):
        self.experiment_id = experiment_id
        self.row_data = row_data
        self.row_id = row_id
        self.result = result
        self.evaluations: dict = {}  # each evaluator's output

    def _write(self, writer: 'ExperimentResultWriter'):
        """Write this ExperimentResult to the writer."""
        writer._write(self)
    
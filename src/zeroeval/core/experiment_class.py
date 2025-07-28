import inspect
import traceback
from typing import Any, Callable, Optional

from zeroeval.observability.tracer import tracer

from .dataset_class import Dataset
from .evaluator_class import Evaluation, Evaluator
from .writer import ExperimentResultBackendWriter, ExperimentResultWriter


class Experiment:
    """
    Represents an experiment that can run a 'task' (the user's function)
    on each row of a dataset, and then evaluate results with 'evaluators'.
    """

    def __init__(
        self, 
        dataset: Dataset, 
        task: Callable[[Any], Any], 
        evaluators: Optional[list[Callable[[Any, Any], Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.dataset = dataset
        self.task = task
        self.evaluators = evaluators or []
        self._evaluator_objs = []

        # If user didn't provide a name, try using the function's __name__
        self.name = name or (task.__name__ if hasattr(task, "__name__") else "unnamed_experiment")

        # If user didn't provide a description, fall back to docstring of task (if any)
        self.description = description or (task.__doc__ or "")

        # We'll treat the default writer as a backend writer.
        self._writer: ExperimentResultWriter = ExperimentResultBackendWriter()
        self.results: list[ExperimentResult] = []

        # Will be set once the experiment is persisted to the backend
        self._backend_id: Optional[str] = None

        # New attribute storing whether we should trace the task calls
        self.trace_task = True

    def run_task(self, subset: Optional[list[dict]] = None, raise_on_error: bool = False) -> list['ExperimentResult']:
        """
        Run the task function on each row (either a given subset or the entire dataset).
        Store the output in self.results and automatically write each result to the backend.
        """
        # Write (or ensure writing) of the Experiment to the backend, retrieving an experiment_id
        experiment_id = self._write(self._writer)  # sets self._backend_id internally
        if not experiment_id:
            return []

        # If subset is given, we assume it includes 'row_id' if needed
        rows_to_run = subset if subset is not None else self.dataset._get_all_full_rows()
        self.results = []

        for row_data in rows_to_run:
            row_id = row_data.get("row_id") if isinstance(row_data, dict) else None
            # "data" sub-dict or entire row
            row_content = row_data["data"] if isinstance(row_data, dict) and "data" in row_data else row_data

            # If tracing is enabled, wrap the task in a span. 
            if self.trace_task:
                from zeroeval.observability.decorators import span
                with span(name=f"experiment:{self.name}") as current_span:
                    try:
                        task_output = self.task(row_content)
                    except Exception as e:
                        if raise_on_error:
                            raise e
                        else:
                            task_output = None
                            # Properly capture error details
                            current_span.set_error(
                                code=e.__class__.__name__,
                                message=str(e),
                                stack=traceback.format_exc()
                            )
                trace_id = current_span.trace_id
            else:
                task_output = self.task(row_content)
                trace_id = None

            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                row_data=row_data,
                row_id=row_id,
                result=task_output,
                trace_id=trace_id,  # <-- Pass the captured trace ID along
                run_number=1  # Default to run 1 for legacy Experiment class
            )
            experiment_result._write(self._writer)
            self.results.append(experiment_result)

        return self.results

    def run_evaluators(
        self, 
        evaluators: Optional[list[Callable[[Any, Any], Any]]] = None, 
        results: Optional[list['ExperimentResult']] = None
    ) -> list['ExperimentResult']:
        """
        Run the specified evaluators on a list of results (or on self.results if none provided).
        Each evaluator is a function: evaluator(row_data, result).
        """
        if evaluators is None:
            evaluators = self.evaluators
        if results is None:
            results = self.results

        
        for evaluator in evaluators:
            evaluator_obj = Evaluator(evaluator.__name__, inspect.getsource(evaluator), evaluator.__doc__, self._backend_id)
            evaluator_obj._write()

            for experiment_result in results:

                row_data_for_eval = (
                    experiment_result.row_data["data"]
                    if isinstance(experiment_result.row_data, dict)
                    and "data" in experiment_result.row_data
                    else experiment_result.row_data
                )

                dataset_row_id = experiment_result.row_id

                evaluation_output = evaluator(row_data_for_eval, experiment_result.result)
                evaluation = Evaluation(evaluator_obj, evaluation_output, experiment_result._backend_id, dataset_row_id)
                evaluation._write()

        return results

    def run(self, subset: Optional[list[dict]] = None) -> list['ExperimentResult']:
        """
        Run tasks and evaluators together, evaluating each task result immediately.
        """
        # Write experiment to backend first
        experiment_id = self._write(self._writer)

        if not experiment_id:
            return []

        # Initialize evaluator objects once
        evaluator_objects = []
        for evaluator in self.evaluators:
            evaluator_obj = Evaluator(
                evaluator.__name__, 
                inspect.getsource(evaluator), 
                evaluator.__doc__, 
                self._backend_id
            )
            evaluator_obj._write()
            evaluator_objects.append(evaluator_obj)

        # Process rows one at a time
        rows_to_run = subset if subset is not None else self.dataset._get_all_full_rows()
        self.results = []

        for row_data in rows_to_run:
            # Run task for this row
            row_id = row_data.get("row_id") if isinstance(row_data, dict) else None
            row_content = row_data["data"] if isinstance(row_data, dict) and "data" in row_data else row_data

            # Run task with tracing if enabled
            if self.trace_task:
                result, trace_id = self._run_traced_task(row_content)
            else:
                result = self.task(row_content)
                trace_id = None

            # Create and write experiment result
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                row_data=row_data,
                row_id=row_id,
                result=result,
                trace_id=trace_id,
                run_number=1  # Default to run 1 for legacy Experiment class
            )
            experiment_result._write(self._writer)
            self.results.append(experiment_result)

            # Immediately run evaluators on this result
            row_data_for_eval = (
                row_data["data"] if isinstance(row_data, dict) and "data" in row_data 
                else row_data
            )
            
            for evaluator, evaluator_obj in zip(self.evaluators, evaluator_objects):
                evaluation_output = evaluator(row_data_for_eval, result)
                evaluation = Evaluation(
                    evaluator_obj, 
                    evaluation_output, 
                    experiment_result._backend_id, 
                    row_id
                )
                evaluation._write()

        return self.results

    def _run_traced_task(self, row_content: Any) -> tuple[Any, Optional[str]]:
        """Helper method to run a task with tracing enabled."""
        from zeroeval.observability.decorators import span
        with span(name=f"experiment:{self.name}") as current_span:
            try:
                task_output = self.task(row_content)
                current_span.set_io(input_data=row_content, output_data=task_output)
            except Exception as e:
                task_output = None
                current_span.set_io(input_data=row_content, output_data=None)
                current_span.set_error(
                    code=e.__class__.__name__,
                    message=str(e),
                    stack=traceback.format_exc()
                )
        tracer.flush()
        return task_output, current_span.trace_id

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
        result: Any,
        trace_id: Optional[str] = None,  # <-- Store the trace ID
        run_number: int = 1  # <-- Add run_number with default
    ):
        self.experiment_id = experiment_id
        self.row_data = row_data
        self.row_id = row_id
        self.result = result
        self._backend_id = None

        # New field to hold the trace ID if tracing was enabled
        self.trace_id = trace_id
        
        # Run number for multiple runs support
        self.run_number = run_number

    def _write(self, writer: 'ExperimentResultWriter'):
        """Write this ExperimentResult to the writer."""
        self._backend_id = writer._write(self)

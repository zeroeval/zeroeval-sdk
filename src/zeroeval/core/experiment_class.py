from typing import List, Callable, Any, Optional
from .dataset_class import Dataset
from .writer import ExperimentResultBackendWriter, ExperimentResultWriter
import inspect

class Experiment:
    def __init__(self, dataset: Dataset, task: Callable, evaluators: List[Callable]):
        self.dataset = dataset
        self.task = task
        self.evaluators = evaluators
        # We'll treat the default writer as a console writer.
        # You can replace this with a backend writer as needed.
        self._writer: ExperimentResultWriter = ExperimentResultBackendWriter()
        self.results: List['ExperimentResult'] = []
        self.task = Task(self.task)

    def run_task(self, subset: Optional[List[dict]] = None) -> List['ExperimentResult']:
        """
        Run the task on each row from the given subset or the entire dataset,
        and store the output in self.results.

        If the row contains a 'row_id', we pass it to the ExperimentResult.
        If the row has the structure {'row_id': <...>, 'data': <columns>},
        we only pass the columns to the task function. The row_id is stored
        separately in the result object.
        """
        experiment_id = self._write(self._writer)
        
        # If subset is given, we assume caller passes full rows with row_id if needed.
        # Otherwise, we fetch full rows from the dataset so we can see row_id.
        if subset is not None:
            rows_to_run = subset
        else:
            rows_to_run = self.dataset._get_all_full_rows()

        self.results = []

        for row_data in rows_to_run:
            possible_row_id = row_data.get("row_id", None)
            row_content = row_data["data"] if "data" in row_data else row_data

            task_output = self.task(row_content)
            experiment_result = ExperimentResult(
                result=task_output,
                experiment_id=experiment_id,
                row_data=row_data,
                row_id=possible_row_id,
                task=self.task
            )
            experiment_result._write(self._writer)
            self.results.append(experiment_result)

        return self.results

    def run_evaluators(
        self, 
        evaluators: Optional[List[Callable]] = None, 
        results: Optional[List['ExperimentResult']] = None
    ) -> List['ExperimentResult']:
        """
        Run the specified evaluators on the provided results or on self.results 
        if no results are passed in, and store the evaluation outputs 
        in each ExperimentResult.

        We only pass the row's "data" portion to each evaluator, 
        excluding the row_id.
        """
        if evaluators is None:
            evaluators = self.evaluators
        if results is None:
            results = self.results

        for experiment_result in results:
            experiment_result.evaluations = {}
            # Extract just the row's "data" portion if present
            row_data_for_eval = (
                experiment_result.row_data["data"] 
                if isinstance(experiment_result.row_data, dict) 
                   and "data" in experiment_result.row_data 
                else experiment_result.row_data
            )

            for evaluator in evaluators:
                evaluation_output = evaluator(row_data_for_eval, experiment_result.result)
                # We store each evaluator's output under its function name
                experiment_result.evaluations[evaluator.__name__] = evaluation_output

        return results

    def run(self, subset: Optional[List[dict]] = None) -> List['ExperimentResult']:
        """
        Run the entire experiment pipeline: tasks, then evaluators.
        Returns the list of results (with their corresponding evaluations).
        """
        self.run_task(subset=subset)
        self.run_evaluators()
        return self.results

    def _write(self, writer: 'ExperimentResultWriter') -> str:
        """
        Write the experiment metadata to the writer. Return the experiment_id assigned by the writer.
        """
        return writer._write(self)


class Task:
    def __init__(self, function: Callable):
        self.function = function
        self.description = function.__doc__ or ""
        self.name = function.__name__
        self.code = inspect.getsource(function)
        self.line_number = inspect.getsourcelines(function)[1]
        self.parameters = function.__code__.co_varnames[:function.__code__.co_argcount]
        self.return_type = function.__annotations__.get("return", None)
        self.return_description = (
            self.return_type.__doc__ if self.return_type and hasattr(self.return_type, "__doc__") else None
        )
    
    def __call__(self, *args, **kwargs):
        """
        Invoke the task function with provided arguments.
        """
        return self.function(*args, **kwargs)

    def __repr__(self):
        return (
            f"Task(name={self.name}, description={self.description}, code={self.code}, "
            f"line_number={self.line_number}, parameters={self.parameters}, "
            f"return_type={self.return_type}, return_description={self.return_description})"
        )


class ExperimentResult:
    """
    Represents the result of a single row's execution under an Experiment. 
    We add 'row_id' so we can associate this result with the dataset row 
    if the dataset is from the backend.
    """
    def __init__(
        self, 
        result: Any, 
        experiment_id: Optional[str] = None, 
        row_data: Optional[dict] = None, 
        row_id: Optional[str] = None,
        task: Optional['Task'] = None
    ):
        self.result = result
        self.experiment_id = experiment_id
        self.row_data = row_data
        self.row_id = row_id
        self.task = task
        self.evaluations = {}

    def _write(self, writer: 'ExperimentResultWriter'):
        """
        Write the experiment result to the selected writer.
        """
        writer._write(self)
    
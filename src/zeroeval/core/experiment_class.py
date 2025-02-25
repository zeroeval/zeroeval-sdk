from typing import List, Callable, Any
from .dataset_class import Dataset
from .writer import ExperimentResultConsoleWriter


class Experiment:
    def __init__(self, dataset: Dataset, task: Callable, evaluators: List[Callable]):
        self.dataset = dataset
        self.task = task
        self.evaluators = evaluators
        self._writer = ExperimentResultConsoleWriter()

    def run(self):
        # Iterate over the dataset and run the task and evaluators
        for data in self.dataset:
            result = self.task(data)
            experiment_result = ExperimentResult(result)
            self._writer.write(experiment_result)
        return result

class ExperimentResult:
    def __init__(self, result: Any):
        self.result = result

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .experiment_class import Experiment, ExperimentResult
    from .evaluator_class import Evaluator, Evaluation

import json
import requests
import os
import logging

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"


class DatasetWriter(ABC):
    """Interface for writing datasets to different destinations."""
    
    @abstractmethod
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """Write a dataset to the destination."""
        pass


class ExperimentResultWriter(ABC):
    """
    Interface for writing experiments and their individual results.
    
    We'll unify both the Experiment and the row-level ExperimentResult
    in this single interface by exposing two different '_write' methods
    based on the passed object type (Experiment vs. ExperimentResult).
    """

    @abstractmethod
    def _write(self, experiment_or_result: Union["Experiment", "ExperimentResult"]) -> Union[str, None]:
        """
        Write an experiment or a single experiment result.
        Return a str (experiment_id) if writing an Experiment,
        or None if writing an ExperimentResult.
        """
        pass


class ExperimentResultConsoleWriter(ExperimentResultWriter):
    """
    Writes experiment and experiment results to the console for debugging.
    """
    def _write(self, experiment_or_result: Union["Experiment", "ExperimentResult"]) -> Union[str, None]:
        from .experiment_class import Experiment, ExperimentResult

        if isinstance(experiment_or_result, Experiment):
            # Writing the experiment itself
            exp = experiment_or_result
            logger.info("Creating experiment for dataset='%s'", exp.dataset.name)
            logger.info("Name: %s", exp.name)
            logger.info("Description: %s", exp.description)
            logger.info("Evaluators: %s", [e.__name__ for e in exp.evaluators])
            # Return a dummy experiment ID for demonstration
            dummy_experiment_id = "console_experiment_id_123"
            logger.info("Assigned experiment_id = %s", dummy_experiment_id)
            return dummy_experiment_id

        elif isinstance(experiment_or_result, ExperimentResult):
            # Writing the experiment result
            res = experiment_or_result
            logger.info("Writing result for experiment_id=%s", res.experiment_id)
            logger.info("row_id=%s, result=%s", res.row_id, res.result)
            logger.info("evaluations=%s", json.dumps(res.evaluations, indent=2))
            return None


class ExperimentResultBackendWriter(ExperimentResultWriter):
    """
    A refined backend writer that:
    1. Resolves workspace name -> workspace ID,
    2. Creates the Experiment,
    3. Creates the Task,
    4. Sends results to the backend.
    """

    def __init__(self) -> None:
        self.api_url = API_URL.rstrip('/')
        self.workspace_name = os.environ.get("WORKSPACE_NAME")
        self._workspace_id = None

    def _get_or_resolve_workspace_id(self) -> Union[str, None]:
        if self._workspace_id:
            return self._workspace_id
        
        if not self.workspace_name:
            logger.warning("No WORKSPACE_NAME was provided. Cannot resolve workspace ID.")
            return None
        
        endpoint = f"{self.api_url}/workspaces/resolve"
        params = {"name": self.workspace_name}
        try:
            resp = requests.get(endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()
            self._workspace_id = data.get("id")
            return self._workspace_id
        except requests.RequestException as exc:
            logger.error("Failed to resolve workspace ID from name '%s': %s", self.workspace_name, exc)
            return None

    def _write(self, experiment_or_result: Union["Experiment", "ExperimentResult"]) -> Union[str, None]:
        from .experiment_class import Experiment, ExperimentResult

        if isinstance(experiment_or_result, Experiment):
            experiment = experiment_or_result
            workspace_id = self._get_or_resolve_workspace_id()
            dataset_version_id = getattr(experiment.dataset, "_version_id", None)

            if not workspace_id or not dataset_version_id:
                logger.error("Missing required workspace or dataset version info. Cannot create experiment.")
                return None

            exp_payload = {
                "workspace_id": workspace_id,
                "dataset_version_id": dataset_version_id,
                "name": experiment.name,
                "description": experiment.description or "",
                "parameters": experiment.parameters
            }
            try:
                exp_response = requests.post(f"{self.api_url}/experiments", json=exp_payload)
                exp_response.raise_for_status()
                exp_data = exp_response.json()
                backend_experiment_id = exp_data["id"]
                logger.info("Created experiment in backend with ID %s.", backend_experiment_id)
                experiment._backend_id = backend_experiment_id

                return backend_experiment_id
            except requests.RequestException as exc:
                logger.error("Failed to create experiment: %s", exc)
                return None

        elif isinstance(experiment_or_result, ExperimentResult):
            res = experiment_or_result

            if not getattr(res, "experiment_id", None):
                logger.error("No experiment_id found in result. Cannot POST this result.")
                return None

            endpoint = f"{self.api_url}/experiments/{res.experiment_id}/results"
            payload = {
                "dataset_row_id": res.row_id or "",
                "result": str(res.result),
                "result_type": "text",
                "trace_id": res.trace_id if res.trace_id else ""
            }
            try:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                logger.info("Successfully posted result for row_id=%s to %s.", res.row_id, endpoint)
                return response.json()["id"]
            except requests.RequestException as exc:
                logger.error("Failed to post result for row_id=%s: %s", res.row_id, exc)
                logger.debug(json.dumps(payload, indent=2))

            return None


class DatasetConsoleWriter(DatasetWriter):
    """Writes datasets to the console for debugging, without showing row_id."""
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """Print dataset to the console, but hide the row_id field."""
        try:
            logger.info("Dataset: %s", dataset.name)
            logger.info("Rows:")
            for row in dataset.rows:
                if isinstance(row, dict):
                    # Hide row_id from display
                    display_row = {k: v for k, v in row.items() if k != "row_id"}
                    logger.info(json.dumps(display_row, indent=2))
                else:
                    logger.info(row)
        except Exception as e:
            logger.error("Error writing dataset to console: %s", e)
            raise


class DatasetBackendWriter(DatasetWriter):
    """Writes datasets to the ZeroEval backend API."""
    
    def __init__(self) -> None:
        self.api_url = API_URL.rstrip('/')
        self.workspace_name = os.environ.get("WORKSPACE_NAME")
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """Write a dataset to the ZeroEval backend."""
        try:
            if not self.workspace_name:
                raise ValueError("WORKSPACE_NAME environment variable must be set")
            
            if create_new_version:
                dataset_id = self._find_existing_dataset_id(dataset.name)
                if dataset_id:
                    self._post_data_to_existing_dataset(dataset_id, dataset)
                    return
            
            # Create a new dataset
            payload = {
                "name": dataset.name,
                "description": dataset.description or "",
                "workspace_name": self.workspace_name
            }
            try:
                response = requests.post(f"{self.api_url}/datasets", json=payload)
                response.raise_for_status()
                dataset_data = response.json()
                dataset_id = dataset_data["id"]
                self._post_data_to_existing_dataset(dataset_id, dataset)
            except requests.RequestException as e:
                self._handle_http_error(e)
                raise RuntimeError("Failed to create dataset") from e
        except Exception as e:
            logger.error("Error writing dataset: %s", e)
            raise

    def _find_existing_dataset_id(self, dataset_name: str) -> Optional[str]:
        """Find dataset ID by name in the workspace."""
        try:
            response = requests.get(f"{self.api_url}/datasets")
            response.raise_for_status()
            datasets = response.json()
            
            return next(
                (ds["id"] for ds in datasets if ds["name"] == dataset_name),
                None
            )
        except requests.RequestException as e:
            raise RuntimeError("Failed to lookup existing dataset: %s" % e)

    def _post_data_to_existing_dataset(self, dataset_id: str, dataset: 'Dataset') -> None:
        """Create a new version of the dataset with the given data."""
        data_as_strings = [
            {k: json.dumps(v) if not isinstance(v, str) else v 
             for k, v in row.items()}
            for row in dataset.data
        ]

        response = requests.post(
            f"{self.api_url}/datasets/{dataset_id}/data",
            json={"data": data_as_strings}
        )
        response.raise_for_status()
        
        version_info = response.json()
        dataset._version_id = version_info["id"]
        dataset._version_number = version_info["version_number"]

    def _handle_http_error(self, error: requests.HTTPError) -> None:
        """Handle common HTTP errors with appropriate messages."""
        if error.response.status_code in (401, 403):
            raise ValueError("Authentication error: Check API key/permissions")
        elif error.response.status_code >= 500:
            raise RuntimeError("Backend server error: %s" % error)
        else:
            try:
                detail = error.response.json().get("detail", str(error))
                raise ValueError("API error: %s" % detail)
            except json.JSONDecodeError as e:
                raise ValueError("API error: %s" % error) from e


class EvaluatorWriter(ABC):
    """Interface for writing evaluators and their evaluations."""
    
    @abstractmethod
    def _write(self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]) -> Union[str, None]:
        """
        Write an evaluator or a single evaluation.
        Return a str (evaluator_id) if writing an Evaluator,
        or None if writing an Evaluation.
        """
        pass


class EvaluatorConsoleWriter(EvaluatorWriter):
    """Writes evaluators and evaluations to the console for debugging."""
    
    def _write(self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]) -> Union[str, None]:
        from .evaluator_class import Evaluator, Evaluation
        
        if isinstance(evaluator_or_evaluation, Evaluator):
            # Writing the evaluator itself
            evaluator = evaluator_or_evaluation
            logger.info("Creating evaluator:")
            logger.info("Name: %s", evaluator.name)
            logger.info("Description: %s", evaluator.description or '[no description]')
            logger.info("Evaluation Type: %s", evaluator.evaluation_type)
            dummy_evaluator_id = f"console_evaluator_id_{evaluator.name}"
            logger.info("Assigned evaluator_id = %s", dummy_evaluator_id)
            return dummy_evaluator_id
            
        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Writing the evaluation result
            evaluation = evaluator_or_evaluation
            logger.info("Writing evaluation:")
            logger.info("Evaluator: %s", evaluation.evaluator.name)
            logger.info("Result: %s", evaluation.result)
            logger.info("Experiment Result ID: %s", evaluation.experiment_result_id)
            logger.info("Dataset Row ID: %s", evaluation.dataset_row_id)
            return None


class EvaluatorBackendWriter(EvaluatorWriter):
    """Writes evaluators and evaluations to the ZeroEval backend."""
    
    def __init__(self) -> None:
        self.api_url = API_URL.rstrip('/')
    
    def _write(self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]) -> Union[str, None]:
        """Write an evaluator or evaluation to the backend."""
        from .evaluator_class import Evaluator, Evaluation

        if isinstance(evaluator_or_evaluation, Evaluator):
            evaluator = evaluator_or_evaluation
            if not evaluator.experiment_id:
                logger.warning("No experiment_id found in evaluator. Cannot create.")
                return None

            payload = {
                "experiment_id": evaluator.experiment_id,  # Required by schema
                "name": evaluator.name,
                "description": evaluator.description or "",
                "code": evaluator.code or ""  # Required by schema, use code instead of evaluator_type
            }
            
            try:
                endpoint = f"{self.api_url}/experiments/{evaluator.experiment_id}/evaluators"
                print(endpoint)
                print(payload)
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                evaluator_data = response.json()
                logger.info("Created evaluator with ID %s", evaluator_data['id'])
                return evaluator_data["id"]
            except requests.RequestException as exc:
                logger.error("Failed to create evaluator: %s", exc)
                return None

        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Create evaluation
            evaluation = evaluator_or_evaluation
            if not evaluation.experiment_result_id:
                logger.warning("Missing experiment_result_id. Cannot create evaluation.")
                return None
            
            experiment_id = evaluation.experiment_result_id.split("_")[0] if evaluation.experiment_result_id else None
            evaluator_id = evaluation.evaluator._backend_id

            if not experiment_id:
                logger.warning("Could not extract experiment_id from result ID. Cannot create evaluation.")
                return None
            
            if not evaluator_id:
                logger.warning("Evaluator has no backend ID. Cannot create evaluation.")
                return None

            payload = {
                "evaluator_id": evaluator_id,
                "dataset_row_id": evaluation.dataset_row_id or "",
                "experiment_result_id": evaluation.experiment_result_id,
                "evaluation_value": str(evaluation.result),
                "evaluation_type": "text"  # Default to text type
            }

            try:
                endpoint = f"{self.api_url}/experiments/{experiment_id}/evaluators/{evaluator_id}/evaluations"
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                evaluation_data = response.json()
                logger.info("Created evaluation with ID %s", evaluation_data['id'])
                return evaluation_data["id"]
            except requests.RequestException as exc:
                logger.error("Failed to create evaluation: %s", exc)
                return None
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union
import json
import requests
import os
if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .experiment_class import Experiment, ExperimentResult
    from .evaluator_class import Evaluator, Evaluation

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
            print(f"[ConsoleWriter] Creating experiment for dataset='{exp.dataset.name}'")
            print(f" - Name: {exp.name}")
            print(f" - Code snippet:\n{exp.code or '[no code]'}")
            print(f" - Description: {exp.description}")
            print(f" - Evaluators: {[e.__name__ for e in exp.evaluators]}")
            # Return a dummy experiment ID for demonstration
            dummy_experiment_id = "console_experiment_id_123"
            print(f"Assigned experiment_id = {dummy_experiment_id}")
            return dummy_experiment_id

        elif isinstance(experiment_or_result, ExperimentResult):
            # Writing the experiment result
            res = experiment_or_result
            print(f"[ConsoleWriter] Writing result for experiment_id={res.experiment_id}")
            print(f"  row_id={res.row_id}, result={res.result}")
            print(f"  evaluations={json.dumps(res.evaluations, indent=2)}")
            return None


class ExperimentResultBackendWriter(ExperimentResultWriter):
    """
    A refined backend writer that:
    1. Resolves workspace name -> workspace ID,
    2. Creates the Experiment,
    3. Creates the Task,
    4. Sends results to the backend.
    """

    def __init__(self):
        self.api_url = API_URL.rstrip('/')
        # If you were previously using WORKSPACE_ID, swap to WORKSPACE_NAME
        self.workspace_name = os.environ.get("WORKSPACE_NAME")
        self._workspace_id = None  # We'll fetch once via the new route

    def _get_or_resolve_workspace_id(self) -> Union[str, None]:
        """
        If we haven't already, call the new route `/workspaces/resolve` with
        the workspace_name to get the workspace ID (provided the user is a member).
        """
        if self._workspace_id:  # Already resolved
            return self._workspace_id
        
        if not self.workspace_name:
            print("[BackendWriter] No WORKSPACE_NAME was provided. Cannot resolve workspace ID.")
            return None
        
        # Attempt to resolve workspace ID from name
        endpoint = f"{self.api_url}/workspaces/resolve"
        params = {"name": self.workspace_name}
        try:
            resp = requests.get(endpoint, params=params)  # Provide auth as needed
            resp.raise_for_status()
            data = resp.json()
            self._workspace_id = data.get("id")
            return self._workspace_id
        except requests.RequestException as exc:
            print(f"[BackendWriter] Failed to resolve workspace ID from name '{self.workspace_name}': {exc}")
            return None

    def _write(self, experiment_or_result: Union["Experiment", "ExperimentResult"]) -> Union[str, None]:
        from .experiment_class import Experiment, ExperimentResult

        if isinstance(experiment_or_result, Experiment):
            experiment = experiment_or_result
            workspace_id = self._get_or_resolve_workspace_id()
            dataset_version_id = getattr(experiment.dataset, "_version_id", None)

            if not workspace_id or not dataset_version_id:
                print("[BackendWriter] Missing required workspace or dataset version info. Cannot create experiment.")
                return None

            # 1. Create the experiment
            exp_payload = {
                "workspace_id": workspace_id,
                "dataset_version_id": dataset_version_id,
                "name": experiment.name,
                "code": experiment.code or "",
                "description": experiment.description or ""
            }
            try:
                exp_response = requests.post(f"{self.api_url}/experiments", json=exp_payload)
                exp_response.raise_for_status()
                exp_data = exp_response.json()
                backend_experiment_id = exp_data["id"]
                print(f"[BackendWriter] Created experiment in backend with ID {backend_experiment_id}.")
                experiment._backend_id = backend_experiment_id

                return backend_experiment_id
            except requests.RequestException as exc:
                print(f"[BackendWriter] Failed to create experiment: {exc}")
                return None

        elif isinstance(experiment_or_result, ExperimentResult):
            # 3. Send the result - Updated payload to match TaskResultCreate schema
            res = experiment_or_result

            if not getattr(res, "experiment_id", None):
                print("[BackendWriter] No experiment_id found in result. Cannot POST this result.")
                return None

            endpoint = f"{self.api_url}/experiments/{res.experiment_id}/results"
            payload = {
                "dataset_row_id": res.row_id or "",
                "result": str(res.result),
                "result_type": "text"  # Using text as default type
            }
            try:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                print(f"[BackendWriter] Successfully posted result for row_id={res.row_id} to {endpoint}.")
                return response.json()["id"]
            except requests.RequestException as exc:
                print(json.dumps(payload, indent=2))
                print(f"[BackendWriter] Failed to post result for row_id={res.row_id}: {exc}")

            return None


class DatasetConsoleWriter(DatasetWriter):
    """Writes datasets to the console for debugging, without showing row_id."""
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """
        Print dataset to the console, but hide the row_id field 
        so it's never displayed to the user.
        """
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
            # If the record has 'row_id', remove or mask it from display
            if isinstance(record, dict) and "row_id" in record:
                display_record = {
                    key: val for key, val in record.items() if key != "row_id"
                }
                print(f"  {i+1}: {display_record}")
            else:
                print(f"  {i+1}: {record}")


class DatasetBackendWriter(DatasetWriter):
    """Writes datasets to the ZeroEval backend API."""
    
    def __init__(self):
        self.api_url = API_URL.rstrip('/')
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """
        Write a dataset to the ZeroEval backend.
        
        Args:
            dataset: The Dataset object to write
            create_new_version: If True, create a new version if dataset exists
        
        Raises:
            ValueError: If dataset already exists or workspace issues
            RuntimeError: If API request fails
        """
        create_url = f"{self.api_url}/datasets/"
        create_payload = {
            "workspace_name": os.environ.get("WORKSPACE_NAME"),
            "name": dataset.name,
            "description": dataset.description or ""
        }

        try:
            response = requests.post(create_url, json=create_payload)
            
            if response.status_code == 409 and create_new_version:
                # Dataset exists - create new version
                existing_id = self._find_existing_dataset_id(dataset.name)
                if not existing_id:
                    raise ValueError("Dataset conflict but not found. Check workspace name.")
                self._post_data_to_existing_dataset(existing_id, dataset)
                dataset._backend_id = existing_id
            elif response.status_code == 409:
                raise ValueError(f"Dataset '{dataset.name}' already exists")
            elif response.status_code == 404:
                raise ValueError("Workspace not found or no access")
            else:
                response.raise_for_status()
                dataset_info = response.json()
                dataset._backend_id = dataset_info["id"]
                self._post_data_to_existing_dataset(dataset._backend_id, dataset)

            print(f"Dataset '{dataset.name}' successfully pushed to ZeroEval.")
            if dataset.version_number:
                print(f"New version number is {dataset.version_number}.")

        except requests.HTTPError as e:
            self._handle_http_error(e)
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

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
            raise RuntimeError(f"Failed to lookup existing dataset: {str(e)}")

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
            raise RuntimeError(f"Backend server error: {str(error)}")
        else:
            try:
                detail = error.response.json().get("detail", str(error))
                raise ValueError(f"API error: {detail}")
            except:
                raise ValueError(f"API error: {str(error)}")


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
            print(f"[ConsoleWriter] Creating evaluator:")
            print(f" - Name: {evaluator.name}")
            print(f" - Code snippet:\n{evaluator.code or '[no code]'}")
            print(f" - Description: {evaluator.description or '[no description]'}")
            # Return a dummy evaluator ID for demonstration
            dummy_evaluator_id = f"console_evaluator_id_{evaluator.name}"
            print(f"Assigned evaluator_id = {dummy_evaluator_id}")
            return dummy_evaluator_id
            
        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Writing the evaluation result
            evaluation = evaluator_or_evaluation
            print(f"[ConsoleWriter] Writing evaluation:")
            print(f" - Evaluator: {evaluation.evaluator.name}")
            print(f" - Result: {evaluation.result}")
            print(f" - Experiment Result ID: {evaluation.experiment_result_id}")
            print(f" - Dataset Row ID: {evaluation.dataset_row_id}")
            return None


class EvaluatorBackendWriter(EvaluatorWriter):
    """Writes evaluators and evaluations to the ZeroEval backend."""
    
    def __init__(self):
        self.api_url = API_URL.rstrip('/')
    
    def _write(self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]) -> Union[str, None]:
        """Write an evaluator or evaluation to the backend."""
        from .evaluator_class import Evaluator, Evaluation

        if isinstance(evaluator_or_evaluation, Evaluator):
            # Create evaluator
            payload = {
                "experiment_id": evaluator_or_evaluation.experiment_id,
                "name": evaluator_or_evaluation.name,
                "code": evaluator_or_evaluation.code,
                "description": evaluator_or_evaluation.description
            }
            
            try:
                endpoint = f"{self.api_url}/experiments/{evaluator_or_evaluation.experiment_id}/evaluators"
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                evaluator_data = response.json()
                print(f"[BackendWriter] Created evaluator with ID {evaluator_data['id']}")
                return evaluator_data["id"]
            except requests.RequestException as exc:
                print(f"[BackendWriter] Failed to create evaluator: {exc}")
                return None

        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Create evaluation
            evaluation = evaluator_or_evaluation
            experiment_id = evaluation.experiment_result_id.split("_")[0]  # Assuming ID format
            evaluator_id = evaluation.evaluator._backend_id
            
            if not evaluator_id:
                print("[BackendWriter] Evaluator has no backend ID. Cannot create evaluation.")
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
                print(f"[BackendWriter] Created evaluation with ID {evaluation_data['id']}")
                return evaluation_data["id"]
            except requests.RequestException as exc:
                print(f"[BackendWriter] Failed to create evaluation: {exc}")
                return None
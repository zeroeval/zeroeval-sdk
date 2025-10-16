import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union
import logging

import requests

if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .evaluator_class import Evaluation, Evaluator
    from .experiment_class import Experiment, ExperimentResult


# Default to production API; environment variables are read lazily at runtime


class DatasetWriter(ABC):
    """Interface for writing datasets to different destinations."""

    @abstractmethod
    def write(self, dataset: "Dataset", create_new_version: bool = False) -> None:
        """Write a dataset to the destination."""
        pass


class ExperimentResultWriter(ABC):
    """
    Interface for writing experiments and their individual results.
    """

    @abstractmethod
    def _write(
        self, experiment_or_result: Union["Experiment", "ExperimentResult"]
    ) -> Union[str, None]:
        """
        Write an experiment or a single experiment result.
        Return a str (experiment_id) if writing an Experiment,
        or None if writing an ExperimentResult.
        """
        pass


class EvaluatorWriter(ABC):
    """Interface for writing evaluators and their evaluations."""

    @abstractmethod
    def _write(
        self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]
    ) -> Union[str, None]:
        """
        Write an evaluator or a single evaluation.
        Return a str (evaluator_id) if writing an Evaluator,
        or None if writing an Evaluation.
        """
        pass


class _BackendWriter:
    """Base class for backend writers to handle authentication."""

    def __init__(self):
        self._api_key: Optional[str] = None
        self._headers: Optional[dict[str, str]] = None

    @property
    def api_url(self):
        """Lazily get API URL from environment to ensure it's read after load_dotenv()"""
        return os.environ.get("ZEROEVAL_API_URL", "https://api.zeroeval.com").rstrip("/")

    def _ensure_auth_setup(self):
        """Ensure API key is resolved and headers are set."""
        if self._api_key is None:
            self._api_key = os.environ.get("ZEROEVAL_API_KEY")
            if not self._api_key:
                raise ValueError("ZEROEVAL_API_KEY environment variable not set")

        if self._headers is None:
            self._headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }


class ExperimentResultBackendWriter(_BackendWriter, ExperimentResultWriter):
    """Sends experiments and results to the backend."""

    def __init__(self):
        super().__init__()

    def _write(
        self, experiment_or_result: Union["Experiment", "ExperimentResult"]
    ) -> Union[str, None]:
        from .experiment_class import Experiment, ExperimentResult

        print(f"[DEBUG] ExperimentResultBackendWriter._write called with {type(experiment_or_result)}")
        
        self._ensure_auth_setup()
        print(f"[DEBUG] Auth setup complete. api_url={self.api_url}")

        if isinstance(experiment_or_result, Experiment):
            experiment = experiment_or_result
            dataset_version_id = getattr(experiment.dataset, "_version_id", None)
            
            print(f"[DEBUG] Experiment write - name='{experiment.name}', dataset_version_id={dataset_version_id}")

            if not dataset_version_id:
                print(f"[ERROR] Missing dataset_version_id: {dataset_version_id}")
                print(f"[DEBUG] Dataset object: {experiment.dataset}")
                print(f"[DEBUG] Dataset attributes: {dir(experiment.dataset)}")
                return None

            exp_payload = {
                "dataset_version_id": dataset_version_id,
                "name": experiment.name,
                "description": experiment.description or "",
            }
            
            print(f"[DEBUG] Sending experiment payload: {exp_payload}")
            
            try:
                exp_response = requests.post(
                    f"{self.api_url}/v1/experiments",
                    json=exp_payload,
                    headers=self._headers,
                )
                print(f"[DEBUG] Experiment POST response status: {exp_response.status_code}")
                print(f"[DEBUG] Experiment POST response text: {exp_response.text}")
                
                exp_response.raise_for_status()
                exp_data = exp_response.json()
                backend_experiment_id = exp_data["id"]
                experiment._backend_id = backend_experiment_id
                print(f"[SUCCESS] Created experiment with ID: {backend_experiment_id}")
                return backend_experiment_id
            except requests.RequestException as e:
                print(f"[ERROR] Failed to create experiment: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"[ERROR] Response status: {e.response.status_code}")
                    print(f"[ERROR] Response text: {e.response.text}")
                return None

        elif isinstance(experiment_or_result, ExperimentResult):
            res = experiment_or_result
            print(f"[DEBUG] ExperimentResult write - experiment_id={getattr(res, 'experiment_id', None)}")

            if not getattr(res, "experiment_id", None):
                print(f"[ERROR] ExperimentResult missing experiment_id")
                return None

            endpoint = f"{self.api_url}/v1/experiments/{res.experiment_id}/results"
            payload = {
                "dataset_row_id": res.row_id or "",
                "result": str(res.result),
                "result_type": "text",
                "trace_id": res.trace_id if res.trace_id else "",
                "run_number": getattr(res, "run_number", 1),  # Default to 1 for backwards compatibility
            }
            
            print(f"[DEBUG] Sending result payload to {endpoint}: {payload}")
            
            try:
                response = requests.post(endpoint, json=payload, headers=self._headers)
                print(f"[DEBUG] Result POST response status: {response.status_code}")
                print(f"[DEBUG] Result POST response text: {response.text}")
                
                response.raise_for_status()
                result_id = response.json()["id"]
                print(f"[SUCCESS] Created experiment result with ID: {result_id}")
                return result_id
            except requests.RequestException as e:
                print(f"[ERROR] Failed to create experiment result: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"[ERROR] Response status: {e.response.status_code}")
                    print(f"[ERROR] Response text: {e.response.text}")
                return None

            return None


class DatasetBackendWriter(_BackendWriter, DatasetWriter):
    """Writes datasets to the ZeroEval backend API using v1 endpoints."""

    def __init__(self):
        super().__init__()
        # Override to use v1 API which doesn't need workspace resolution
        self._use_v1_api = True

    def _ensure_auth_setup(self):
        """Ensure API key is set for v1 API."""
        if self._api_key is None:
            self._api_key = os.environ.get("ZEROEVAL_API_KEY")
            if not self._api_key:
                raise ValueError("ZEROEVAL_API_KEY environment variable not set")

        if self._headers is None:
            self._headers = {"Authorization": f"Bearer {self._api_key}"}

    def write(self, dataset: "Dataset", create_new_version: bool = False) -> None:
        print(f"[DEBUG] DatasetBackendWriter.write called for dataset '{dataset.name}'")
        self._ensure_auth_setup()

        # Use v1 API endpoint
        create_url = f"{self.api_url}/v1/datasets"
        create_payload = {
            "name": dataset.name,
            "description": dataset.description or "",
        }
        
        print(f"[DEBUG] Dataset create payload: {create_payload}")

        try:
            response = requests.post(
                create_url, json=create_payload, headers=self._headers
            )
            
            print(f"[DEBUG] Dataset create response status: {response.status_code}")
            print(f"[DEBUG] Dataset create response text: {response.text}")

            if response.status_code == 409:
                # Dataset already exists, add data to it
                print(f"[DEBUG] Dataset already exists, adding data to existing dataset")
                self._post_data_to_existing_dataset_v1(dataset.name, dataset)
            elif response.status_code == 404:
                raise ValueError("Workspace not found or no access")
            else:
                response.raise_for_status()
                dataset_info = response.json()
                dataset._backend_id = dataset_info["id"]
                print(f"[DEBUG] Created new dataset with ID: {dataset._backend_id}")
                self._post_data_to_existing_dataset_v1(dataset.name, dataset)

        except requests.HTTPError as e:
            print(f"[ERROR] HTTP error in dataset write: {e}")
            self._handle_http_error(e)
        except requests.RequestException as e:
            print(f"[ERROR] Request error in dataset write: {e}")
            raise RuntimeError(f"Connection error: {str(e)}")

    def _post_data_to_existing_dataset_v1(
        self, dataset_name: str, dataset: "Dataset"
    ) -> None:
        """Create a new version of the dataset with the given data using v1 API."""
        print(f"[DEBUG] _post_data_to_existing_dataset_v1 called for dataset '{dataset_name}'")
        self._ensure_auth_setup()
        data_as_strings = [
            {k: json.dumps(v) if not isinstance(v, str) else v for k, v in row.items()}
            for row in dataset.data
        ]
        
        print(f"[DEBUG] Posting {len(data_as_strings)} rows to dataset")

        response = requests.post(
            f"{self.api_url}/v1/datasets/{dataset_name}/data",
            json={"data": data_as_strings},
            headers=self._headers,
        )
        
        print(f"[DEBUG] Dataset data POST response status: {response.status_code}")
        print(f"[DEBUG] Dataset data POST response text: {response.text}")
        
        response.raise_for_status()

        version_info = response.json()
        dataset._version_id = version_info["id"]
        dataset._version_number = version_info["version_number"]
        
        print(f"[SUCCESS] Dataset version created - ID: {dataset._version_id}, version: {dataset._version_number}")

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


class EvaluatorBackendWriter(_BackendWriter, EvaluatorWriter):
    """Writes evaluators and evaluations to the ZeroEval backend."""

    def __init__(self):
        super().__init__()

    def _write(
        self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]
    ) -> Union[str, None]:
        """Write an evaluator or evaluation to the backend."""
        from .evaluator_class import Evaluation, Evaluator

        print(f"[DEBUG] EvaluatorBackendWriter._write called with {type(evaluator_or_evaluation)}")
        
        self._ensure_auth_setup()

        if isinstance(evaluator_or_evaluation, Evaluator):
            # Create evaluator
            payload = {
                "experiment_id": evaluator_or_evaluation.experiment_id,
                "name": evaluator_or_evaluation.name,
                "description": evaluator_or_evaluation.description,
                "evaluation_mode": evaluator_or_evaluation.evaluation_mode,
            }

            print(f"[DEBUG] Creating evaluator with payload: {payload}")

            try:
                endpoint = f"{self.api_url}/v1/experiments/{evaluator_or_evaluation.experiment_id}/evaluators"
                response = requests.post(endpoint, json=payload, headers=self._headers)
                
                print(f"[DEBUG] Evaluator creation response status: {response.status_code}")
                print(f"[DEBUG] Evaluator creation response text: {response.text}")
                
                response.raise_for_status()
                evaluator_data = response.json()
                evaluator_id = evaluator_data["id"]
                print(f"[SUCCESS] Created evaluator with ID: {evaluator_id}")
                return evaluator_id
            except requests.RequestException as e:
                print(f"[ERROR] Failed to create evaluator: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"[ERROR] Response status: {e.response.status_code}")
                    print(f"[ERROR] Response text: {e.response.text}")
                return None

        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Create evaluation
            print(f"[DEBUG] EvaluatorBackendWriter._write - processing Evaluation object")
            evaluation = evaluator_or_evaluation
            experiment_id = (
                evaluation.evaluator.experiment_id
            )  # Get experiment ID from evaluator
            evaluator_id = evaluation.evaluator._backend_id

            print(f"[DEBUG] Creating evaluation - experiment_id={experiment_id}, evaluator_id={evaluator_id}")
            print(f"[DEBUG] Evaluator object: {evaluation.evaluator}")
            print(f"[DEBUG] Evaluator._backend_id: {evaluation.evaluator._backend_id}")

            if not evaluator_id:
                print(f"[ERROR] Cannot create evaluation - missing evaluator_id (evaluator_id={evaluator_id})")
                return None

            payload = {
                "evaluator_id": evaluator_id,
                "dataset_row_id": evaluation.dataset_row_id or "",
                "experiment_result_id": evaluation.experiment_result_id,
                "evaluation_value": str(evaluation.result),
                "evaluation_type": "text",  # Default to text type
            }

            print(f"[DEBUG] Creating evaluation with payload: {payload}")

            try:
                endpoint = f"{self.api_url}/v1/experiments/{experiment_id}/evaluations"
                response = requests.post(endpoint, json=payload, headers=self._headers)
                
                print(f"[DEBUG] Evaluation creation response status: {response.status_code}")
                print(f"[DEBUG] Evaluation creation response text: {response.text}")
                
                response.raise_for_status()
                evaluation_data = response.json()
                evaluation_id = evaluation_data["id"]
                print(f"[SUCCESS] Created evaluation with ID: {evaluation_id}")
                return evaluation_id
            except requests.RequestException as e:
                print(f"[ERROR] Failed to create evaluation: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"[ERROR] Response status: {e.response.status_code}")
                    print(f"[ERROR] Response text: {e.response.text}")
                return None
        
        else:
            print(f"[ERROR] EvaluatorBackendWriter._write called with unknown type: {type(evaluator_or_evaluation)}")
            return None
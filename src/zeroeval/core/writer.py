from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union
import json
import requests
import os

if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .experiment_class import Experiment, ExperimentResult
    from .evaluator_class import Evaluator, Evaluation


# Default to production API; for local dev set BACKEND_URL env var to "http://localhost:8000" (or another URL)
API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


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
        self.api_url = os.environ.get("ZEROEVAL_API_URL", "http://localhost:8000").rstrip("/")
        self._api_key: Optional[str] = None
        self._workspace_id: Optional[str] = None
        self._headers: Optional[Dict[str, str]] = None

    def _ensure_auth_setup(self):
        """Ensure API key and workspace ID are resolved and headers are set."""
        if self._api_key is None:
            self._api_key = os.environ.get("ZEROEVAL_API_KEY")
            if not self._api_key:
                raise ValueError("ZEROEVAL_API_KEY environment variable not set")

        if self._workspace_id is None:
            try:
                response = requests.post(
                    f"{self.api_url}/api-keys/resolve", json={"api_key": self._api_key}
                )
                response.raise_for_status()
                response_data = response.json()

                if "workspace_id" not in response_data:
                    raise ValueError("API key does not resolve to a workspace")

                self._workspace_id = response_data["workspace_id"]

            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    raise ValueError("Invalid API key")
                elif e.response.status_code == 404:
                    raise ValueError("API key does not resolve to a workspace")
                else:
                    raise ValueError(
                        f"Failed to resolve API key (HTTP {e.response.status_code})"
                    )
            except requests.RequestException as e:
                raise RuntimeError(f"Network error while resolving API key: {str(e)}")

        if self._headers is None:
            self._headers = {"Authorization": f"Bearer {self._api_key}"}


class ExperimentResultBackendWriter(_BackendWriter, ExperimentResultWriter):
    """Sends experiments and results to the backend."""

    def __init__(self):
        super().__init__()

    def _write(
        self, experiment_or_result: Union["Experiment", "ExperimentResult"]
    ) -> Union[str, None]:
        from .experiment_class import Experiment, ExperimentResult

        self._ensure_auth_setup()

        if isinstance(experiment_or_result, Experiment):
            experiment = experiment_or_result
            dataset_version_id = getattr(experiment.dataset, "_version_id", None)

            if not self._workspace_id or not dataset_version_id:
                return None

            exp_payload = {
                "workspace_id": self._workspace_id,
                "dataset_version_id": dataset_version_id,
                "name": experiment.name,
                "description": experiment.description or "",
            }
            try:
                exp_response = requests.post(
                    f"{self.api_url}/workspaces/{self._workspace_id}/experiments",
                    json=exp_payload,
                    headers=self._headers,
                )
                exp_response.raise_for_status()
                exp_data = exp_response.json()
                backend_experiment_id = exp_data["id"]
                experiment._backend_id = backend_experiment_id
                return backend_experiment_id
            except requests.RequestException:
                return None

        elif isinstance(experiment_or_result, ExperimentResult):
            res = experiment_or_result

            if not getattr(res, "experiment_id", None):
                return None

            endpoint = f"{self.api_url}/workspaces/{self._workspace_id}/experiments/{res.experiment_id}/results"
            payload = {
                "dataset_row_id": res.row_id or "",
                "result": str(res.result),
                "result_type": "text",
                "trace_id": res.trace_id if res.trace_id else "",
            }
            try:
                response = requests.post(endpoint, json=payload, headers=self._headers)
                response.raise_for_status()
                return response.json()["id"]
            except requests.RequestException:
                pass

            return None


class DatasetBackendWriter(_BackendWriter, DatasetWriter):
    """Writes datasets to the ZeroEval backend API."""

    def __init__(self):
        super().__init__()

    def write(self, dataset: "Dataset", create_new_version: bool = False) -> None:
        self._ensure_auth_setup()

        create_url = f"{self.api_url}/workspaces/{self._workspace_id}/datasets"
        create_payload = {
            "workspace_name": os.environ.get("ZEROEVAL_WORKSPACE_NAME"),
            "name": dataset.name,
            "description": dataset.description or "",
        }

        try:
            response = requests.post(
                create_url, json=create_payload, headers=self._headers
            )

            if response.status_code == 409 and create_new_version:
                existing_id = self._find_existing_dataset_id(dataset.name)
                if not existing_id:
                    raise ValueError(
                        "Dataset conflict but not found. Check workspace name."
                    )
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

        except requests.HTTPError as e:
            self._handle_http_error(e)
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")

    def _find_existing_dataset_id(self, dataset_name: str) -> Optional[str]:
        """Find dataset ID by name in the workspace."""
        self._ensure_auth_setup()
        try:
            response = requests.get(
                f"{self.api_url}/workspaces/{self._workspace_id}/datasets",
                headers=self._headers,
            )
            response.raise_for_status()
            datasets = response.json()

            return next(
                (ds["id"] for ds in datasets if ds["name"] == dataset_name), None
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to lookup existing dataset: {str(e)}")

    def _post_data_to_existing_dataset(
        self, dataset_id: str, dataset: "Dataset"
    ) -> None:
        """Create a new version of the dataset with the given data."""
        self._ensure_auth_setup()
        data_as_strings = [
            {k: json.dumps(v) if not isinstance(v, str) else v for k, v in row.items()}
            for row in dataset.data
        ]

        response = requests.post(
            f"{self.api_url}/workspaces/{self._workspace_id}/datasets/{dataset_id}/data",
            json={"data": data_as_strings},
            headers=self._headers,
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


class EvaluatorBackendWriter(_BackendWriter, EvaluatorWriter):
    """Writes evaluators and evaluations to the ZeroEval backend."""

    def __init__(self):
        super().__init__()

    def _write(
        self, evaluator_or_evaluation: Union["Evaluator", "Evaluation"]
    ) -> Union[str, None]:
        """Write an evaluator or evaluation to the backend."""
        from .evaluator_class import Evaluator, Evaluation

        self._ensure_auth_setup()

        if isinstance(evaluator_or_evaluation, Evaluator):
            # Create evaluator
            payload = {
                "experiment_id": evaluator_or_evaluation.experiment_id,
                "name": evaluator_or_evaluation.name,
                "description": evaluator_or_evaluation.description,
            }

            try:
                endpoint = f"{self.api_url}/workspaces/{self._workspace_id}/experiments/{evaluator_or_evaluation.experiment_id}/evaluators"
                response = requests.post(endpoint, json=payload, headers=self._headers)
                response.raise_for_status()
                evaluator_data = response.json()
                return evaluator_data["id"]
            except requests.RequestException:
                return None

        elif isinstance(evaluator_or_evaluation, Evaluation):
            # Create evaluation
            evaluation = evaluator_or_evaluation
            experiment_id = (
                evaluation.evaluator.experiment_id
            )  # Get experiment ID from evaluator
            evaluator_id = evaluation.evaluator._backend_id

            if not evaluator_id:
                return None

            payload = {
                "evaluator_id": evaluator_id,
                "dataset_row_id": evaluation.dataset_row_id or "",
                "experiment_result_id": evaluation.experiment_result_id,
                "evaluation_value": str(evaluation.result),
                "evaluation_type": "text",  # Default to text type
            }

            try:
                endpoint = f"{self.api_url}/workspaces/{self._workspace_id}/experiments/{experiment_id}/evaluators/{evaluator_id}/evaluations"
                response = requests.post(endpoint, json=payload, headers=self._headers)
                response.raise_for_status()
                evaluation_data = response.json()
                return evaluation_data["id"]
            except requests.RequestException:
                return None
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import requests
import os

from .init import _validate_init

if TYPE_CHECKING:
    from .dataset_class import Dataset

API_URL = "http://localhost:8000"


class DatasetReader(ABC):
    """Interface for reading datasets from different sources."""
    
    @abstractmethod
    def pull_by_id(self, dataset_id: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from a destination using its ID.
        """
        pass

    @abstractmethod
    def pull_by_name(self, workspace_id: str, dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from a destination using its workspace ID and dataset name.
        """
        pass


class DatasetBackendReader(DatasetReader):
    """
    Reads datasets from the ZeroEval backend API, analogous to DatasetBackendWriter.
    """
    def __init__(self):
        """
        Initialize with a base URL, falling back to localhost if not set.
        """
        self.base_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    
    def get_workspace_id_by_name(self, workspace_name: str) -> str:
        """
        Query the backend for the workspace by its name, returning the workspace ID.
        """
        url = f"{self.base_url}/workspaces/by_name/{workspace_name}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            workspace = resp.json()
            return workspace["id"]
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch workspace ID by name: {e}")

    def pull_by_id(self, dataset_id: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from the ZeroEval backend by ID.
        
        Args:
            dataset_id: The UUID or string ID of the dataset to fetch.
            version_number: Optional version number to fetch (defaults to latest).
        
        Returns:
            A Dataset instance populated with the fetched data.
        """
        _validate_init()
        
        # Import here to avoid circular import
        from .dataset_class import Dataset
        
        # 1) Fetch dataset metadata
        dataset_info_url = f"{self.base_url}/datasets/{dataset_id}"
        try:
            dataset_info_resp = requests.get(dataset_info_url)
            dataset_info_resp.raise_for_status()
            dataset_info = dataset_info_resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch dataset info by ID: {str(e)}")
        
        # 2) Fetch stored rows
        data_url = f"{self.base_url}/datasets/{dataset_id}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
        
        try:
            data_resp = requests.get(data_url, params=params)
            data_resp.raise_for_status()
            data_json = data_resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch dataset rows: {str(e)}")

        # 3) Build Dataset object 
        dataset = Dataset(
            name=dataset_info["name"],
            data=data_json["rows"],
            description=dataset_info.get("description")
        )

        # 4) Update backend metadata 
        dataset._backend_id = dataset_id
        dataset._version_id = data_json["version"]["id"]
        dataset._version_number = data_json["version"]["version_number"]
        
        return dataset

    def pull_by_name(self, workspace_id: str, dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset by workspace ID and dataset name.
        """
        _validate_init()

        from .dataset_class import Dataset

        info_url = f"{self.base_url}/workspaces/{workspace_id}/datasets/{dataset_name}"
        try:
            info_resp = requests.get(info_url)
            info_resp.raise_for_status()
            dataset_info = info_resp.json()
        except requests.RequestException as e:
            # If we received a 404, raise a more friendly error indicating that the dataset was not found
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found for workspace '{os.environ.get('WORKSPACE_NAME')}'."
                    "Verify that the dataset name and workspace are correct."
                ) from e
            # Otherwise, raise a generic runtime error
            raise RuntimeError(f"Failed to fetch dataset info by name: {e}") from e

        dataset_id = dataset_info["id"]

        # 2) Fetch rows + version
        data_url = f"{self.base_url}/workspaces/{workspace_id}/datasets/{dataset_name}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
        
        try:
            data_resp = requests.get(data_url, params=params)
            data_resp.raise_for_status()
            data_json = data_resp.json()
        except requests.RequestException as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"No data found for dataset '{dataset_name}' in workspace '{workspace_id}' "
                    f"(version: {version_number if version_number else 'latest'})."
                ) from e
            raise RuntimeError(f"Failed to fetch dataset rows by name: {e}") from e
        
        dataset = Dataset(
            name=dataset_info["name"],
            data=data_json["rows"],
            description=dataset_info.get("description")
        )

        dataset._backend_id = dataset_id
        dataset._version_id = data_json["version"]["id"]
        dataset._version_number = data_json["version"]["version_number"]
        return dataset

    def pull_by_workspace_name(
        self, workspace_name: str, dataset_name: str, version_number: Optional[int] = None
    ):
        """
        Fetch dataset metadata and rows from the backend by workspace_name and dataset_name.
        """
        # 1) Get dataset metadata
        meta_url = f"{self.base_url}/workspaces/name/{workspace_name}/datasets/{dataset_name}"
        r_meta = requests.get(meta_url)
        r_meta.raise_for_status()
        dataset_meta = r_meta.json()

        # 2) Get dataset data
        data_url = f"{self.base_url}/workspaces/name/{workspace_name}/datasets/{dataset_name}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number

        r_data = requests.get(data_url, params=params)
        r_data.raise_for_status()
        dataset_data = r_data.json()

        # Build local Dataset object from the pulled data
        return self._build_dataset_from_response(dataset_meta, dataset_data)

    def _build_dataset_from_response(self, dataset_meta, dataset_data):
        """
        Helper to construct a local Dataset instance (or dictionary).
        """
        # Your logic to parse dataset_meta and dataset_data as needed.
        ...
        # For demonstration, return a dict:
        return {
            "id": dataset_meta["id"],
            "workspace_id": dataset_meta["workspace_id"],
            "name": dataset_meta["name"],
            "description": dataset_meta["description"],
            "columns": dataset_data["columns"],
            "rows": dataset_data["rows"],
            "version": dataset_data["version"]["version_number"],
        }

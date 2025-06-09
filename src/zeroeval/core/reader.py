from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import requests
import os

from .init import _validate_init

if TYPE_CHECKING:
    from .dataset_class import Dataset



class DatasetReader(ABC):
    """Interface for reading datasets from different sources."""
    
    @abstractmethod
    def pull_by_id(self, dataset_id: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from a destination using its ID.
        """
        pass

    @abstractmethod
    def pull_by_name(self, workspace_id: Optional[str], dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from a destination using its dataset name.
        Note: workspace_id may be ignored if resolved from API key.
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
        self.base_url = os.environ.get("ZEROEVAL_API_URL", "http://localhost:8000")
        self._api_key = None
        self._workspace_id = None
        self._headers = None
    
    def _ensure_auth_setup(self):
        """Ensure API key and workspace ID are resolved and headers are set."""
        if self._api_key is None:
            self._api_key = os.environ.get("ZEROEVAL_API_KEY")
            if not self._api_key:
                raise ValueError("ZEROEVAL_API_KEY environment variable not set")
        
        if self._workspace_id is None:
            try:
                response = requests.post(
                    f"{self.base_url}/api-keys/resolve", 
                    json={"api_key": self._api_key}
                )
                response.raise_for_status()
                response_data = response.json()
                
                if "workspace_id" not in response_data:
                    raise ValueError(f"API key does not resolve to a workspace")
                
                self._workspace_id = response_data["workspace_id"]
                
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    raise ValueError(f"Invalid API key")
                elif e.response.status_code == 404:
                    raise ValueError(f"API key does not resolve to a workspace")
                else:
                    raise ValueError(f"Failed to resolve API key (HTTP {e.response.status_code})")
            except requests.RequestException as e:
                raise RuntimeError(f"Network error while resolving API key: {str(e)}")
        
        if self._headers is None:
            self._headers = {"Authorization": f"Bearer {self._api_key}"}
    
    def get_workspace_id_by_name(self, workspace_name: str) -> str:
        """
        Query the backend for the workspace by its name, returning the workspace ID.
        """
        self._ensure_auth_setup()
        url = f"{self.base_url}/workspaces/by_name/{workspace_name}"
        try:
            resp = requests.get(url, headers=self._headers)
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
        self._ensure_auth_setup()
        
        # Import here to avoid circular import
        from .dataset_class import Dataset
        
        # 1) Fetch dataset metadata
        dataset_info_url = f"{self.base_url}/workspaces/{self._workspace_id}/datasets/{dataset_id}"
        try:
            dataset_info_resp = requests.get(dataset_info_url, headers=self._headers)
            dataset_info_resp.raise_for_status()
            dataset_info = dataset_info_resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch dataset info by ID: {str(e)}")
        
        # 2) Fetch stored rows
        data_url = f"{self.base_url}/workspaces/{self._workspace_id}/datasets/{dataset_id}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
        
        try:
            data_resp = requests.get(data_url, params=params, headers=self._headers)
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

    def pull_by_name(self, workspace_id: Optional[str], dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset by dataset name.
        Note: workspace_id parameter is ignored as workspace is resolved from API key.
        """
        _validate_init()
        self._ensure_auth_setup()

        from .dataset_class import Dataset

        info_url = f"{self.base_url}/workspaces/{self._workspace_id}/datasets/{dataset_name}"
        try:
            info_resp = requests.get(info_url, headers=self._headers)
            info_resp.raise_for_status()
            dataset_info = info_resp.json()
        except requests.RequestException as e:
            # If we received a 404, raise a more friendly error indicating that the dataset was not found
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in your workspace. "
                    "Verify that the dataset name is correct."
                ) from e
            # Otherwise, raise a generic runtime error
            raise RuntimeError(f"Failed to fetch dataset info by name: {e}") from e

        dataset_id = dataset_info["id"]

        # 2) Fetch rows + version
        data_url = f"{self.base_url}/workspaces/{self._workspace_id}/datasets/{dataset_name}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
        
        try:
            data_resp = requests.get(data_url, params=params, headers=self._headers)
            data_resp.raise_for_status()
            data_json = data_resp.json()
        except requests.RequestException as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"No data found for dataset '{dataset_name}' in your workspace "
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

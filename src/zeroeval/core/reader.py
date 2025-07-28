import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import requests

from .init import _validate_init

if TYPE_CHECKING:
    from .dataset_class import Dataset



class DatasetReader(ABC):
    """Interface for reading datasets from different sources."""
    
    @abstractmethod
    def pull_by_name(self, dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset from a destination using its dataset name.
        Workspace is automatically resolved from API key.
        """
        pass


class DatasetBackendReader(DatasetReader):
    """
    Reads datasets from the ZeroEval backend API using the new v1 endpoints.
    """
    def __init__(self):
        """
        Initialize with a base URL, falling back to localhost if not set.
        """
        self.base_url = os.environ.get("ZEROEVAL_API_URL", "https://api.zeroeval.com")
        self._api_key = None
        self._headers = None
    
    def _ensure_auth_setup(self):
        """Ensure API key is set and headers are configured."""
        if self._api_key is None:
            self._api_key = os.environ.get("ZEROEVAL_API_KEY")
            if not self._api_key:
                raise ValueError("ZEROEVAL_API_KEY environment variable not set")
        
        if self._headers is None:
            self._headers = {"Authorization": f"Bearer {self._api_key}"}

    def pull_by_name(self, dataset_name: str, version_number: Optional[int] = None) -> "Dataset":
        """
        Pull a dataset by dataset name using the v1 API.
        Workspace is automatically resolved from API key.
        """
        if not _validate_init():
            return None
        self._ensure_auth_setup()

        from .dataset_class import Dataset

        # 1) Fetch dataset metadata using v1 API
        info_url = f"{self.base_url}/v1/datasets/{dataset_name}"
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

        # 2) Fetch rows + version using v1 API
        data_url = f"{self.base_url}/v1/datasets/{dataset_name}/data"
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
        # Set a high limit to get all rows (v1 API supports up to 10000)
        params["limit"] = 10000
        params["offset"] = 0
        
        all_rows = []
        total_rows = None
        
        # Paginate through all rows
        while True:
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
            
            # Collect rows
            all_rows.extend(data_json["rows"])
            
            # Check if we have all rows
            if total_rows is None and "totalRows" in data_json:
                total_rows = data_json["totalRows"]
            
            # If we have all rows or no total_count, break
            if total_rows is None or len(all_rows) >= total_rows:
                break
                
            # Otherwise, fetch next page
            params["offset"] = len(all_rows)
        
        # Create dataset with all rows
        dataset = Dataset(
            dataset_info["name"],
            data=all_rows,
            description=dataset_info.get("description")
        )

        # Set backend metadata
        dataset._backend_id = dataset_id
        dataset._version_id = data_json["version"]["id"]
        dataset._version_number = data_json["version"]["version_number"]
        
        return dataset

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional
import json
import requests
import os
if TYPE_CHECKING:
    from .dataset_class import Dataset
    from .experiment_class import ExperimentResult

API_URL = "http://localhost:8000"


class DatasetWriter(ABC):
    """Interface for writing datasets to different destinations."""
    
    @abstractmethod
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """Write a dataset to the destination."""
        pass


class DatasetConsoleWriter(DatasetWriter):
    """Writes datasets to the console for debugging."""
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """Print dataset to the console in a readable format."""
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
            print(f"  {i+1}: {record}")


class ExperimentResultWriter(ABC):
    """Interface for writing experiment results to different destinations."""
    
    @abstractmethod
    def write(self, experiment_result: 'ExperimentResult') -> None:
        """Write experiment results to the destination."""
        pass


class ExperimentResultConsoleWriter(ExperimentResultWriter):
    """Writes experiment results to the console for debugging."""
    
    def write(self, experiment_result: 'ExperimentResult') -> None:
        """Print experiment results to the console in a readable format."""
        print(f"Writing experiment results to console: {experiment_result.result}")


class DatasetBackendWriter(DatasetWriter):
    """Writes datasets to the ZeroEval backend API."""
    
    def __init__(self):
        """
        Initialize the backend writer with API configuration.
        
        Args:
            api_url: Base URL for the ZeroEval API
            workspace_id: ID of the workspace to use
            user_id: ID of the user performing the operation
        """
        self.api_url = API_URL.rstrip('/')
    
    def write(self, dataset: 'Dataset', create_new_version: bool = False) -> None:
        """
        Write a dataset to the ZeroEval backend.
        
        This method tries to create a new dataset in the backend. If the name 
        already exists and `create_new_version` is True, then it 
        creates a new version of that existing dataset with the given data.
        
        Args:
            dataset: The Dataset object to write
            create_new_version: If True, create a new version in case 
                the dataset already exists. Defaults to False.
        
        Raises:
            RuntimeError: If the API request fails
        """
        # 1) Attempt to create the dataset
        create_url = f"{self.api_url}/datasets/"
        create_payload = {
            "workspace_name": os.environ.get("WORKSPACE_NAME"),
            "name": dataset.name,
            "description": dataset.description or ""
        }

        print(f"Creating dataset: {create_payload}")
        
        try:
            response = requests.post(create_url, json=create_payload)
            
            if response.status_code == 409:
                # "Conflict": dataset with this name already exists
                if create_new_version:
                    # Find the existing dataset ID by name
                    existing_id = self._find_existing_dataset_id(dataset.name)
                    if not existing_id:
                        # If somehow not found, raise
                        raise ValueError(
                            "Dataset name conflict, but existing dataset not found. "
                            "Check your Workspace Name or credentials."
                        )
                    # Create a new version by posting data to /datasets/{existing_id}/data
                    self._post_data_to_existing_dataset(existing_id, dataset)
                    dataset._backend_id = existing_id
                else:
                    # If user does NOT want to create a new version, raise
                    detail = "A dataset with this name already exists in the workspace"
                    try:
                        error_json = response.json()
                        if "detail" in error_json:
                            detail = error_json["detail"]
                    except:
                        pass
                    raise ValueError(f"Dataset name conflict: {detail}")
            
            elif response.status_code == 404:
                # "Workspace not found" or no access
                error_detail = "Workspace not found or you don't have access to it"
                try:
                    error_json = response.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                except:
                    pass
                raise ValueError(f"Workspace error: {error_detail}")
            
            else:
                # For other codes: standard raise_for_status
                response.raise_for_status()
                
                # If created, get the info
                dataset_info = response.json()
                dataset._backend_id = dataset_info["id"]
                
                # 2) Upload the data as version #1
                self._post_data_to_existing_dataset(dataset._backend_id, dataset)
            
            print(f"Dataset '{dataset.name}' successfully pushed to ZeroEval.")
            if dataset.version_number:
                print(f"New version number is {dataset.version_number}.")
            
        except requests.HTTPError as e:
            if e.response.status_code in [401, 403]:
                raise ValueError("Authentication error: Please check your API key or permissions")
            elif e.response.status_code >= 500:
                raise RuntimeError(f"Backend server error: {str(e)}")
            else:
                try:
                    error_detail = e.response.json().get("detail", str(e))
                    raise ValueError(f"API error: {error_detail}")
                except:
                    raise ValueError(f"API error: {str(e)}")
        except requests.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")
        except ValueError:
            # Re-raise ValueError exceptions (our custom ones)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to write dataset to backend: {str(e)}")
    
    def _find_existing_dataset_id(self, dataset_name: str) -> str:
        """
        Query the backend for all datasets, find one with the given name 
        in the same workspace, and return its id. If not found, return None.
        """
        try:
            print(f"Finding existing dataset ID for {dataset_name}")
            url = f"{self.api_url}/datasets"
            resp = requests.get(url)
            print(f"Response: {resp.json()}")
            resp.raise_for_status()
            all_datasets = resp.json()  # list of dataset records
            
            # We rely on the environment's WORKSPACE_NAME to match
            # or trust that the user can only see datasets they have access to.
            for ds in all_datasets:
                if ds["name"] == dataset_name:
                    return ds["id"]
            return None
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to lookup existing dataset: {str(e)}")
    
    def _post_data_to_existing_dataset(self, dataset_id: str, dataset: 'Dataset') -> None:
        """
        Add data to an existing dataset by creating a new version.
        Updates the dataset object with the newly returned version info.
        """
        # Prepare rows so that each value is a string
        data_as_strings = []
        for row in dataset.data:
            row_str_only = {}
            for col_name, col_val in row.items():
                if isinstance(col_val, str):
                    # Already a string
                    row_str_only[col_name] = col_val
                else:
                    # Convert to string (JSON for dicts, lists, etc.)
                    row_str_only[col_name] = json.dumps(col_val)
            data_as_strings.append(row_str_only)

        data_url = f"{self.api_url}/datasets/{dataset_id}/data"
        
        data_payload = {
            "data": data_as_strings
        }
        
        resp = requests.post(data_url, json=data_payload)
        resp.raise_for_status()
        version_info = resp.json()
        
        dataset._version_id = version_info["id"]
        dataset._version_number = version_info["version_number"]
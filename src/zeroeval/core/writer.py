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
    def write(self, dataset: 'Dataset') -> None:
        """Write a dataset to the destination."""
        pass


class DatasetConsoleWriter(DatasetWriter):
    """Writes datasets to the console for debugging."""
    
    def write(self, dataset: 'Dataset') -> None:
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
    def write(self, dataset: 'Dataset') -> None:
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
    
    def write(self, dataset: 'Dataset') -> None:
        """
        Write a dataset to the ZeroEval backend.
        
        This method:
        1. Creates a new dataset in the backend
        2. Uploads the data to the dataset
        3. Updates the dataset object with backend properties
        
        Args:
            dataset: The Dataset object to write
        
        Raises:
            RuntimeError: If the API request fails
        """
        # Step 1: Create the dataset
        create_url = f"{self.api_url}/datasets/"
        create_payload = {
            "workspace_name": os.environ.get("WORKSPACE_NAME"),
            "name": dataset.name,
            "description": dataset.description or ""
        }

        print(f"Creating dataset: {create_payload}")
        
        try:
            response = requests.post(create_url, json=create_payload)
            
            # Handle specific error cases
            if response.status_code == 409:
                error_detail = "A dataset with this name already exists in the workspace"
                try:
                    error_json = response.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                except:
                    pass
                raise ValueError(f"Dataset name conflict: {error_detail}")
            elif response.status_code == 404:
                error_detail = "Workspace not found or you don't have access to it"
                try:
                    error_json = response.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                except:
                    pass
                raise ValueError(f"Workspace error: {error_detail}")
            
            # For other errors, use the standard raise_for_status
            response.raise_for_status()
            dataset_info = response.json()
            
            # Update the dataset with backend ID
            dataset._backend_id = dataset_info["id"]
            
            # Step 2: Upload the data
            data_url = f"{self.api_url}/datasets/{dataset._backend_id}/data"
            data_payload = {
                "data": dataset.data
            }
            
            response = requests.post(data_url, json=data_payload)
            response.raise_for_status()
            version_info = response.json()
            
            # Update the dataset with version info
            dataset._version_id = version_info["id"]
            dataset._version_number = version_info["version_number"]
            
            print(f"Dataset '{dataset.name}' successfully uploaded to ZeroEval backend")
            print(f"Dataset ID: {dataset._backend_id}, Version: {dataset._version_number}")
            
        except requests.HTTPError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                raise ValueError("Authentication error: Please check your API key or permissions")
            elif e.response.status_code >= 500:
                raise RuntimeError(f"Backend server error: {str(e)}")
            else:
                # Try to extract more detailed error message from response
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
from typing import TYPE_CHECKING, List, Dict, Any, Optional
from .writer import DatasetBackendWriter
from .reader import DatasetBackendReader

from .init import _validate_init
import os
if TYPE_CHECKING:
    from .writer import DatasetWriter

class Dataset:
    """
    A class to represent a named collection of dictionary records.
    
    Attributes:
        name (str): The name of the dataset
        data (list): A list of dictionaries containing the data
        description (str): A description of the dataset
        backend_id (str): The ID of the dataset in the backend (after pushing)
        version_id (str): The ID of the dataset version in the backend
        version_number (int): The version number in the backend
    """
    
    def __init__(self, name: str, data: List[Dict[str, Any]], description: Optional[str] = None):
        """
        Initialize a Dataset with a name and data.
        
        Args:
            name (str): The name of the dataset
            data (list): A list of dictionaries containing the data
            description (str): A description of the dataset
            
        Raises:
            TypeError: If name is not a string or data is not a list
            ValueError: If any item in data is not a dictionary
        """
        if not isinstance(name, str):
            raise TypeError("Dataset name must be a string")
        
        if not isinstance(data, list):
            raise TypeError("Dataset data must be a list")
            
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("All items in data must be dictionaries")
            
        self._name = name
        self._data = data.copy()  # Create a copy to avoid external modifications
        self._description = description
        self._writer = DatasetBackendWriter()
        
        # Backend properties (set after pushing to backend)
        self._backend_id = None
        self._version_id = None
        self._version_number = None

    def add_rows(self, new_rows: List[Dict[str, Any]]) -> None:
        """
        Add one or more rows to the dataset.
        
        Args:
            new_rows (List[Dict[str, Any]]): The list of data rows to add.
            
        Raises:
            TypeError: If new_rows isn't a list or if any row isn't a dictionary.
        """
        if not isinstance(new_rows, list):
            raise TypeError("new_rows must be a list of dictionaries.")
        if not all(isinstance(row, dict) for row in new_rows):
            raise TypeError("All items in new_rows must be dictionaries.")
        
        self._data.extend(new_rows)

    def delete_row(self, index: int) -> None:
        """
        Delete a row from the dataset by index.
        
        Args:
            index (int): The index of the row to delete.
            
        Raises:
            IndexError: If the index is out of range.
        """
        try:
            del self._data[index]
        except IndexError:
            raise IndexError(f"Cannot delete row {index}, index is out of range.")

    def update_row(self, index: int, new_data: Dict[str, Any]) -> None:
        """
        Update a single row in the dataset by index, replacing its entire contents.
        
        Args:
            index (int): The index of the row to update.
            new_data (Dict[str, Any]): The new dictionary data for this row.
        
        Raises:
            TypeError: If new_data is not a dictionary.
            IndexError: If the index is out of range.
        """
        self.__setitem__(index, new_data)

    def push(self, writer=None, create_new_version: bool = False):
        """
        Push the dataset to a storage destination.
        
        Args:
            writer: Optional writer to use. If None, the default writer is used.
            create_new_version (bool): If True, and a dataset with this name already exists, 
                a new version will be created instead of failing. Defaults to False.
                
        Returns:
            self: Returns self for method chaining
        """
        _validate_init()
        if writer:
            writer.write(self, create_new_version=create_new_version)
        else:
            self._writer.write(self, create_new_version=create_new_version)
        return self

    @classmethod
    def pull(
        cls,
        dataset_name: str,
        version_number: Optional[int] = None,
        workspace_name: Optional[str] = None
    ) -> "Dataset":
        """
        Pull a dataset by workspace *name* + dataset name, optionally specifying version_number.
        
        This uses the DatasetBackendReader to:
          1) Convert the workspace_name to a workspace_id
          2) Fetch metadata from GET /workspaces/{workspace_id}/datasets/{dataset_name}
          3) Fetch rows from GET /workspaces/{workspace_id}/datasets/{dataset_name}/data
        """
        _validate_init()
        
        # Fall back to environment variable if none given
        if not workspace_name:
            workspace_name = os.environ.get("WORKSPACE_NAME")
            if not workspace_name:
                raise ValueError(
                    "No workspace_name provided, and WORKSPACE_NAME env var is not set."
                )

        reader = DatasetBackendReader()
        # 1) Convert from workspace_name -> workspace_id
        workspace_id = reader.get_workspace_id_by_name(workspace_name)
        
        # 2) Use the existing pull-by-ID approach
        return reader.pull_by_name(workspace_id, dataset_name, version_number=version_number)
    
    @property
    def version_id(self):
        """Get the backend version ID (if pushed)."""
        return self._version_id
    
    @property
    def version_number(self):
        """Get the backend version number (if pushed)."""
        return self._version_number
    
    @property
    def name(self):
        """Get the dataset name."""
        return self._name
    
    @property
    def description(self):
        """Get the dataset description."""
        return self._description

    @property
    def data(self):
        """Get a copy of the dataset data."""
        return self._data.copy()
    
    @property
    def columns(self):
        """Derive columns from the first record keys, ignoring the backend's returned columns."""
        if not self._data:
            return set()
        return set(self._data[0].keys())
    
    def write(self, writer: 'DatasetWriter'):
        """Write the dataset using the provided writer."""
        writer.write(self)
    
    def __len__(self):
        """Return the number of records in the dataset."""
        return len(self._data)
    
    def __getitem__(self, idx: int):
        """Allow indexing to access records."""
        return self._data[idx]
    
    def __setitem__(self, idx: int, value: Dict[str, Any]):
        """
        Allow updating a row at the given index, e.g. dataset[idx] = {…}.
        
        Args:
            idx (int): The index to update.
            value (Dict[str, Any]): The new data for that index.
            
        Raises:
            TypeError: If the new value is not a dictionary.
            IndexError: If idx is out of range.
        """
        if not isinstance(value, dict):
            raise TypeError("Each row must be a dictionary.")
        try:
            self._data[idx] = value
        except IndexError:
            raise IndexError(f"Cannot set row {idx}, index is out of range.")

    def __delitem__(self, idx: int):
        """
        Allow deleting a row at the given index with del dataset[idx].
        
        Args:
            idx (int): The index of the row to delete.
        """
        self.delete_row(idx)

    def __str__(self):
        """String representation of the dataset."""
        return f"Dataset('{self._name}', {len(self._data)} records)"
    
    def __repr__(self):
        """Detailed representation of the dataset."""
        return f"Dataset(name='{self._name}', size={len(self._data)}, columns={self.columns})"

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data)

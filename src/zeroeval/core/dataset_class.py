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
        self._description = description
        # Keep the full rows (possibly including row_id, data, etc.)
        self._data = data.copy()  # avoid external modifications
        
        self._writer = DatasetBackendWriter()
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

    def push(self, create_new_version: bool = False):
        """
        Push the dataset to a storage destination.
        
        Args:
            create_new_version (bool): If True, and a dataset with this name already exists, 
                a new version will be created instead of failing. Defaults to False.
                
        Returns:
            self: Returns self for method chaining
        """
        _validate_init()
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
        Pull a dataset by dataset name, optionally specifying version_number.
        
        This uses the DatasetBackendReader to:
          1) Resolve workspace ID from API key
          2) Fetch metadata and rows from the backend
        """
        _validate_init()

        reader = DatasetBackendReader()
        # The reader will handle workspace resolution from API key internally
        return reader.pull_by_name(None, dataset_name, version_number=version_number)
    
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        By design, return *only* the 'data' part to the user,
        hiding row_id from direct indexing.
        """
        row = self._data[idx]
        # Return the data portion if present, else the row as-is
        return row["data"] if "data" in row else row
    
    def __setitem__(self, idx: int, value: Dict[str, Any]):
        """
        Updating a row at the given index. If the existing row has a row_id, preserve it.
        For direct user usage, the caller is presumably passing the 'data' portion.
        """
        if not isinstance(value, dict):
            raise TypeError("Each row must be a dictionary.")
        try:
            existing_row = self._data[idx]
            # If the existing row has a row_id, keep it
            if "row_id" in existing_row:
                # Wrap the new user data in {"data": value} if needed
                if "data" in value:
                    # They might have used the same structure. We'll trust them.
                    self._data[idx] = {"row_id": existing_row["row_id"], **value}
                else:
                    # They passed only data, so wrap it
                    self._data[idx] = {"row_id": existing_row["row_id"], "data": value}
            else:
                # If there is no row_id, just replace the entire row
                self._data[idx] = value
        except IndexError:
            raise IndexError(f"Cannot set row {idx}, index is out of range.")

    def __iter__(self):
        """
        By design, yield only the 'data' portion for iteration,
        so normal iteration does not expose row_id.
        """
        for row in self._data:
            yield row["data"] if "data" in row else row

    @property
    def data(self) -> List[Dict[str, Any]]:
        """
        Returns a list of just the data portion for each row.
        """
        return [row["data"] if "data" in row else row for row in self._data]

    @property
    def columns(self):
        """
        Derive columns from the first record's data portion if present.
        """
        if not self._data:
            return set()
        first_row = self._data[0]
        data_part = first_row["data"] if "data" in first_row else first_row
        return set(data_part.keys())

    # -------------------
    # Internal API: methods that return full rows.
    # Not intended for typical user usage, but for internal calls.
    # -------------------
    def _get_full_row(self, idx: int) -> Dict[str, Any]:
        """Return the entire row dictionary, including row_id, data, etc."""
        return self._data[idx]

    def _get_all_full_rows(self) -> List[Dict[str, Any]]:
        """Return all row dictionaries in their entirety."""
        return self._data

    def __delitem__(self, idx: int):
        """
        Allow deleting a row at the given index with del dataset[idx].
        """
        self.delete_row(idx)

    def __str__(self):
        """String representation of the dataset."""
        return f"Dataset('{self._name}', {len(self._data)} records)"
    
    def __repr__(self):
        """Detailed representation of the dataset."""
        return f"Dataset(name='{self._name}', size={len(self._data)}, columns={self.columns})"

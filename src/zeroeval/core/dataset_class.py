from typing import TYPE_CHECKING, List, Dict, Any
from .writer import DatasetBackendWriter
import requests
from .init import _validate_init
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
    
    def __init__(self, name, data, description=None):
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
    
    def push(self, writer=None):
        """
        Push dataset to a storage destination.
        
        Args:
            writer: Optional writer to use. If None, uses the default writer.
            
        Returns:
            self: Returns self for method chaining
        """
        _validate_init()
        if writer:
            writer.write(self)
        else:
            self._writer.write(self)
        return self

    @classmethod
    def pull(cls, dataset_id: str, api_url: str, version_number: int = None):
        """
        Pull dataset from the ZeroEval backend.
        
        Args:
            dataset_id: ID of the dataset to pull
            api_url: Base URL for the ZeroEval API
            version_number: Optional specific version to pull (defaults to latest)
            
        Returns:
            Dataset: A new Dataset instance with the pulled data
        """
        _validate_init()
        api_url = api_url.rstrip('/')
        url = f"{api_url}/datasets/{dataset_id}/data"
        
        params = {}
        if version_number is not None:
            params["version_number"] = version_number
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Get dataset info
            dataset_response = requests.get(f"{api_url}/datasets/{dataset_id}")
            dataset_response.raise_for_status()
            dataset_info = dataset_response.json()
            
            # Create the dataset
            dataset = cls(
                name=dataset_info["name"],
                data=data["rows"],
                description=dataset_info.get("description")
            )
            
            # Set backend properties
            dataset._backend_id = dataset_id
            dataset._version_id = data["version"]["id"]
            dataset._version_number = data["version"]["version_number"]
            
            return dataset
            
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to pull dataset from backend: {str(e)}")
    
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
        """Get the column names from the first record, or empty set if no data."""
        if not self._data:
            return set()
        return set(self._data[0].keys())
    
    def write(self, writer: 'DatasetWriter'):
        """Write the dataset using the provided writer."""
        writer.write(self)
    
    def __len__(self):
        """Return the number of records in the dataset."""
        return len(self._data)
    
    def __getitem__(self, index):
        """Allow indexing to access records."""
        return self._data[index]
    
    def __str__(self):
        """String representation of the dataset."""
        return f"Dataset('{self._name}', {len(self._data)} records)"
    
    def __repr__(self):
        """Detailed representation of the dataset."""
        return f"Dataset(name='{self._name}', size={len(self._data)}, columns={self.columns})"

    def __iter__(self):
        """Return an iterator over the dataset."""
        return iter(self._data)

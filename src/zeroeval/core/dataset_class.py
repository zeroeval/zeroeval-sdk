from typing import TYPE_CHECKING, List, Dict, Any
from .writer import DatasetConsoleWriter

if TYPE_CHECKING:
    from .writer import DatasetWriter

class Dataset:
    """
    A class to represent a named collection of dictionary records.
    
    Attributes:
        name (str): The name of the dataset
        data (list): A list of dictionaries containing the data
        description (str): A description of the dataset
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
        self._writer = DatasetConsoleWriter()
    
    def push(self):
        """Push dataset to a remote storage."""
        self._writer.write(self)
    
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

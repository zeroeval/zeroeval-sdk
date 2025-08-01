import base64
import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, Callable
import csv

from .init import _validate_init
from .reader import DatasetBackendReader
from .writer import DatasetBackendWriter

if TYPE_CHECKING:
    from .run import Run


class DotDict(dict):
    """Dictionary that supports dot notation access."""
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value


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
    
    def __init__(self, name_or_path: str, data: Optional[list[dict[str, Any]]] = None, description: Optional[str] = None):
        """
        Initialize a Dataset with a name and data, or load from a CSV file.
        
        Args:
            name_or_path (str): Dataset name OR path to a CSV file
            data (list, optional): List of dictionaries containing the data
            description (str, optional): Description of the dataset
            
        Usage:
            # From CSV file
            ds = Dataset("/path/to/data.csv")
            
            # From list of dicts
            ds = Dataset("my_dataset", data=[...])
            
        Raises:
            TypeError: If name_or_path is not a string
            ValueError: If data format is invalid
        """
        if not isinstance(name_or_path, str):
            raise TypeError("First argument must be a string (dataset name or file path)")
        
        # Check if it's a file path
        if data is None and (name_or_path.endswith('.csv') or os.path.exists(name_or_path)):
            # Load from CSV
            self._name = Path(name_or_path).stem  # Use filename without extension as name
            self._description = description or f"Dataset loaded from {name_or_path}"
            self._data = self._load_csv(name_or_path)
        else:
            # Traditional initialization
            self._name = name_or_path
            self._description = description
            
            if data is None:
                raise ValueError("Must provide data when not loading from a file")
            
            if not isinstance(data, list):
                raise TypeError("Dataset data must be a list")
                
            if not all(isinstance(item, dict) for item in data):
                raise ValueError("All items in data must be dictionaries")
                
            self._data = data.copy()  # avoid external modifications
        
        self._writer = DatasetBackendWriter()
        self._backend_id = None
        self._version_id = None
        self._version_number = None
    
    def _load_csv(self, file_path: str) -> list[dict[str, Any]]:
        """Load data from a CSV file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))  # Convert OrderedDict to regular dict
        return data

    def add_rows(self, new_rows: list[dict[str, Any]]) -> None:
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

    def update_row(self, index: int, new_data: dict[str, Any]) -> None:
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
        
        If a dataset with the same name already exists, a new version will be automatically created.
        
        Args:
            create_new_version (bool): For backward compatibility. This parameter is no longer needed
                as new versions are automatically created when a dataset name already exists.
                
        Returns:
            self: Returns self for method chaining
        """
        if not _validate_init():
            return self
        self._writer.write(self, create_new_version=create_new_version)
        
        # After pushing, refresh the local data with backend data that includes row_ids
        try:
            updated_dataset = Dataset.pull(self.name)
            self._data = updated_dataset._data
        except Exception as e:
            print(f"Warning: Could not refresh dataset data from backend: {e}")
        
        return self
    


    def run(self, task_func: Callable, run_number: int = 1, total_runs: int = 1, experiment_id: Optional[str] = None) -> "Run":
        """
        Run a task function on this dataset without mutating it.
        
        Args:
            task_func: A @task decorated function
            run_number: The run number (for multiple runs)
            total_runs: Total number of runs (for multiple runs)
            experiment_id: Optional experiment ID to share across runs
            
        Returns:
            Run object containing results that can be evaluated
            
        Example:
            @task(outputs=["pred"])
            def solve(row):
                return {"pred": llm_answer(row.question)}
                
            results = dataset.run(solve)
            results.eval([exact_match])
        """
        from .run import Run
        from zeroeval.observability.decorators import span
        import traceback
        
        # Validate that it's a task
        if not hasattr(task_func, '_is_task'):
            raise TypeError(f"{task_func.__name__} is not a @task decorated function")
            
        # Prepare result rows
        result_rows = []
        
        for row_data in self._data:
            # Make a copy to avoid mutating original data
            row_copy = row_data.copy()
            
            # Extract the actual data if nested
            if 'data' in row_copy and isinstance(row_copy['data'], dict):
                # Flatten the structure for easier access
                actual_data = row_copy['data'].copy()
                # Keep row_id if present
                if 'row_id' in row_copy:
                    actual_data['row_id'] = row_copy['row_id']
            else:
                actual_data = row_copy
            
            # Convert to DotDict for dot notation access
            dot_row = DotDict(actual_data)
            
            # Run the task with tracing
            with span(name=f"task:{task_func._task_name}") as current_span:
                try:
                    task_output = task_func(dot_row)
                    # Merge task output with row data
                    actual_data.update(task_output)
                except Exception as e:
                    # Capture error but continue processing
                    current_span.set_error(
                        code=e.__class__.__name__,
                        message=str(e),
                        stack=traceback.format_exc()
                    )
                    actual_data["_error"] = str(e)
                    
            result_rows.append(actual_data)
            
        # Create and return Run object
        return Run(
            dataset_name=self.name,
            dataset_id=getattr(self, "_backend_id", ""),
            dataset_version_id=getattr(self, "_version_id", ""),
            task_name=task_func._task_name,
            task_code=task_func._task_code,
            rows=result_rows,
            outputs=task_func._outputs,
            run_number=run_number,
            total_runs=total_runs,
            task_func=task_func,
            dataset_ref=self,
            experiment_id=experiment_id
        )

    @classmethod
    def pull(
        cls,
        name: str,
        version_number: Optional[int] = None
    ) -> "Dataset":
        """
        Pull a dataset by name, optionally specifying version_number.
        
        The workspace is automatically resolved from your API key.
        
        Args:
            name: Name of the dataset to pull
            version_number: Optional specific version to pull (defaults to latest)
            
        Returns:
            Dataset instance with the pulled data
        """
        if not _validate_init():
            return None

        reader = DatasetBackendReader()
        return reader.pull_by_name(name, version_number=version_number)
    
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
        
    @staticmethod
    def _encode_file_to_base64(file_path: str) -> str:
        """Convert a file to base64 encoding.
        
        Args:
            file_path: Path to the file to encode
            
        Returns:
            Base64 encoded string with appropriate data URI prefix
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Default to octet-stream if can't determine type
            mime_type = 'application/octet-stream'
            
        with open(file_path, 'rb') as file:
            encoded = base64.b64encode(file.read()).decode('utf-8')
            return f'data:{mime_type};base64,{encoded}'
    
    def add_image(self, row_index: int, column_name: str, image_path: str) -> None:
        """Add an image to a specific cell in the dataset.
        
        Args:
            row_index: Index of the row to update
            column_name: Name of the column to add the image to
            image_path: Path to the image file
            
        Raises:
            IndexError: If row_index is out of range
            FileNotFoundError: If image file does not exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check if image path has valid extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        ext = Path(image_path).suffix.lower()
        if ext not in valid_extensions:
            raise ValueError(f"Invalid image file extension: {ext}. Must be one of {valid_extensions}")
            
        # Encode image as base64
        encoded_image = self._encode_file_to_base64(image_path)
        
        # Update cell with the encoded image
        try:
            row = self._data[row_index]
            if "data" in row:
                row["data"][column_name] = encoded_image
            else:
                row[column_name] = encoded_image
        except IndexError:
            raise IndexError(f"Row index {row_index} is out of range")
    
    def add_audio(self, row_index: int, column_name: str, audio_path: str) -> None:
        """Add an audio file to a specific cell in the dataset.
        
        Args:
            row_index: Index of the row to update
            column_name: Name of the column to add the audio to
            audio_path: Path to the audio file
            
        Raises:
            IndexError: If row_index is out of range
            FileNotFoundError: If audio file does not exist
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check if audio path has valid extension
        valid_extensions = ['.mp3', '.wav', '.ogg', '.m4a']
        ext = Path(audio_path).suffix.lower()
        if ext not in valid_extensions:
            raise ValueError(f"Invalid audio file extension: {ext}. Must be one of {valid_extensions}")
            
        # Encode audio as base64
        encoded_audio = self._encode_file_to_base64(audio_path)
        
        # Update cell with the encoded audio
        try:
            row = self._data[row_index]
            if "data" in row:
                row["data"][column_name] = encoded_audio
            else:
                row[column_name] = encoded_audio
        except IndexError:
            raise IndexError(f"Row index {row_index} is out of range")
    
    def add_video(self, row_index: int, column_name: str, video_path: str) -> None:
        """Add a video file to a specific cell in the dataset.
        
        Args:
            row_index: Index of the row to update
            column_name: Name of the column to add the video to
            video_path: Path to the video file
            
        Raises:
            IndexError: If row_index is out of range
            FileNotFoundError: If video file does not exist
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Check if video path has valid extension
        valid_extensions = ['.mp4', '.webm', '.mov']
        ext = Path(video_path).suffix.lower()
        if ext not in valid_extensions:
            raise ValueError(f"Invalid video file extension: {ext}. Must be one of {valid_extensions}")
            
        # Encode video as base64
        encoded_video = self._encode_file_to_base64(video_path)
        
        # Update cell with the encoded video
        try:
            row = self._data[row_index]
            if "data" in row:
                row["data"][column_name] = encoded_video
            else:
                row[column_name] = encoded_video
        except IndexError:
            raise IndexError(f"Row index {row_index} is out of range")
            
    def add_media_url(self, row_index: int, column_name: str, media_url: str, media_type: str) -> None:
        """Add a media URL to a specific cell in the dataset.
        
        Args:
            row_index: Index of the row to update
            column_name: Name of the column to add the media URL to
            media_url: URL pointing to the media file
            media_type: Type of media ('image', 'audio', or 'video')
            
        Raises:
            IndexError: If row_index is out of range
            ValueError: If media_type is invalid
        """
        valid_types = ['image', 'audio', 'video']
        if media_type not in valid_types:
            raise ValueError(f"Invalid media type: {media_type}. Must be one of {valid_types}")
            
        # Update cell with the media URL
        try:
            row = self._data[row_index]
            if "data" in row:
                row["data"][column_name] = media_url
            else:
                row[column_name] = media_url
        except IndexError:
            raise IndexError(f"Row index {row_index} is out of range")

    def __getitem__(self, key: Union[int, slice]) -> Union[DotDict, "Dataset"]:
        """
        Supports indexing and slicing of the dataset.
        
        Args:
            key: An integer index or slice object
            
        Returns:
            - For integer index: A single row as DotDict
            - For slice: A new Dataset with the sliced rows
            
        Examples:
            row = dataset[0]      # Get first row
            subset = dataset[:10] # Get first 10 rows
            subset = dataset[5:]  # Get all rows from index 5
        """
        if isinstance(key, int):
            # Single row access
            return DotDict(self._data[key])
        elif isinstance(key, slice):
            # Slice access - return a new Dataset
            sliced_data = self._data[key]
            subset = Dataset(
                f"{self._name}_slice",
                data=sliced_data,
                description=f"Slice of {self._name}"
            )
            # Preserve backend metadata if available
            subset._backend_id = self._backend_id
            subset._version_id = self._version_id
            subset._version_number = self._version_number
            return subset
        else:
            raise TypeError(f"Dataset indices must be integers or slices, not {type(key).__name__}")
    
    def __setitem__(self, idx: int, value: dict[str, Any]):
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
        Makes the dataset iterable, yielding DotDict instances for each row.
        
        Example:
            for row in dataset:
                print(row.name, row.score)
        """
        for row in self._data:
            yield DotDict(row)
            
    def __len__(self):
        """
        Returns the number of rows in the dataset.
        """
        return len(self._data)

    @property
    def data(self) -> list[dict[str, Any]]:
        """
        Returns a list of just the data portion for each row.
        """
        return [row.get("data", row) for row in self._data]
        
    @property
    def columns(self) -> list[str]:
        """
        Returns a list of column names in the dataset.
        """
        # Collect all unique keys from all rows
        columns = set()
        for row in self._data:
            if "data" in row and isinstance(row["data"], dict):
                columns.update(row["data"].keys())
            else:
                columns.update(row.keys())
                
        # Remove internal keys like 'row_id' if present
        if "row_id" in columns:
            columns.remove("row_id")
        
        return sorted(list(columns))



    # -------------------
    # Internal API: methods that return full rows.
    # Not intended for typical user usage, but for internal calls.
    # -------------------
    def _get_full_row(self, idx: int) -> dict[str, Any]:
        """Return the entire row dictionary, including row_id, data, etc."""
        return self._data[idx]

    def _get_all_full_rows(self) -> list[dict[str, Any]]:
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

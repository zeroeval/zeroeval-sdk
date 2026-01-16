import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Integration(ABC):
    """Base class for all tracing integrations."""
    
    # Required package for this integration
    PACKAGE_NAME: str = None
    
    def __init__(self, tracer):
        self.tracer = tracer
        # Store originals by object identity so teardown can reliably restore,
        # even for dynamically created classes (e.g. LangChain Runnables).
        self._original_functions: dict[tuple[Any, str], Callable] = {}
        self._setup_attempted = False
        self._setup_successful = False
        self._setup_error = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if the required package is installed."""
        if cls.PACKAGE_NAME is None:
            return False
        try:
            importlib.import_module(cls.PACKAGE_NAME)
            return True
        except ImportError:
            return False

    @abstractmethod
    def setup(self) -> None:
        """Setup the integration by applying all necessary patches."""
        pass

    def safe_setup(self) -> bool:
        """Safely attempt to setup the integration, catching and storing any errors."""
        if self._setup_attempted:
            return self._setup_successful
            
        self._setup_attempted = True
        try:
            self.setup()
            self._setup_successful = True
            return True
        except Exception as exc:
            self._setup_error = exc
            self._setup_successful = False
            return False

    def get_setup_error(self) -> Optional[Exception]:
        """Get the error that occurred during setup, if any."""
        return self._setup_error

    def teardown(self) -> None:
        """Teardown the integration by removing all patches."""
        for (target_object, method_name), original_func in list(self._original_functions.items()):
            try:
                setattr(target_object, method_name, original_func)
            except Exception:
                pass
        self._original_functions.clear()

    def _patch_method(self, target_object: Any, method_name: str, wrapper: Callable) -> None:
        """Helper method to patch an object's method."""
        original = getattr(target_object, method_name)

        # Skip if already patched by ZeroEval
        if getattr(original, "__ze_patched__", False):
            return

        self._original_functions[(target_object, method_name)] = original

        patched = wrapper(original)
        # Mark so we can recognise it later and avoid double wrapping
        patched.__ze_patched__ = True

        setattr(target_object, method_name, patched)

    def _unpatch_method(self, target_object: Any, method_name: str) -> None:
        """Helper method to restore an object's original method."""
        key = (target_object, method_name)
        if key in self._original_functions:
            setattr(target_object, method_name, self._original_functions[key])
            del self._original_functions[key]

    def _get_object_by_path(self, obj_path: str) -> Any:
        """Helper to get an object by its module path."""
        module_path, obj_name = obj_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, obj_name)
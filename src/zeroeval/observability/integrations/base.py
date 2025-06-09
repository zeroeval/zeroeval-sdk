from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type
import importlib

class Integration(ABC):
    """Base class for all tracing integrations."""
    
    # Required package for this integration
    PACKAGE_NAME: str = None
    
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

    def __init__(self, tracer):
        self.tracer = tracer
        self._original_functions: Dict[str, Callable] = {}

    @abstractmethod
    def setup(self) -> None:
        """Setup the integration by applying all necessary patches."""
        pass

    def teardown(self) -> None:
        """Teardown the integration by removing all patches."""
        for key, original_func in self._original_functions.items():
            obj_name, method_name = key.rsplit('.', 1)
            try:
                obj = self._get_object_by_path(obj_name)
                setattr(obj, method_name, original_func)
            except:
                pass
        self._original_functions.clear()

    def _patch_method(self, target_object: Any, method_name: str, wrapper: Callable) -> None:
        """Helper method to patch an object's method."""
        original = getattr(target_object, method_name)

        # Skip if already patched by ZeroEval
        if getattr(original, "__ze_patched__", False):
            return

        # Identify the patched object name in a readable / unique way.
        if isinstance(target_object, type):
            obj_name = target_object.__name__  # class name
        else:
            obj_name = target_object.__class__.__name__  # instance name

        self._original_functions[f"{obj_name}.{method_name}"] = original

        patched = wrapper(original)
        # Mark so we can recognise it later and avoid double wrapping
        setattr(patched, "__ze_patched__", True)

        setattr(target_object, method_name, patched)

    def _unpatch_method(self, target_object: Any, method_name: str) -> None:
        """Helper method to restore an object's original method."""
        key = f"{target_object.__class__.__name__}.{method_name}"
        if key in self._original_functions:
            setattr(target_object, method_name, self._original_functions[key])
            del self._original_functions[key]
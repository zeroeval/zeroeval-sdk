"""Python version and dependency compatibility tests."""

from importlib import import_module
from types import ModuleType

import pytest


class TestPythonVersionCompatibility:
    """Test compatibility across Python versions."""

    @pytest.mark.compatibility
    def test_python_version_supported(self, python_version):
        """Test that current Python version is supported."""
        major, minor = python_version.major, python_version.minor

        # SDK requires Python 3.9+
        assert major == 3, f"Python {major}.{minor} not supported, need Python 3.x"
        assert minor >= 9, f"Python 3.{minor} not supported, need Python 3.9+"

    @pytest.mark.compatibility
    def test_python_version_info(self, python_version):
        """Test Python version info structure."""
        assert hasattr(python_version, "major")
        assert hasattr(python_version, "minor")
        assert hasattr(python_version, "micro")
        assert isinstance(python_version.major, int)
        assert isinstance(python_version.minor, int)
        assert isinstance(python_version.micro, int)

    @pytest.mark.compatibility
    def test_standard_library_imports(self):
        """Test that required standard library modules are available."""
        required_modules = [
            "json",
            "os",
            "sys",
            "typing",
            "asyncio",
            "pathlib",
            "datetime",
            "uuid",
            "threading",
        ]

        for module_name in required_modules:
            try:
                module = import_module(module_name)
                assert isinstance(module, ModuleType)
            except ImportError:
                pytest.fail(
                    f"Required standard library module '{module_name}' not available"
                )


class TestDependencyCompatibility:
    """Test compatibility with required dependencies."""

    @pytest.mark.compatibility
    def test_core_dependencies_importable(self):
        """Test that all core dependencies can be imported."""
        core_deps = [
            "rich",
            "opentelemetry",
            "requests",
            "openai",
            "PIL",  # Pillow
            "aiohttp",
        ]

        for dep in core_deps:
            try:
                import_module(dep)
            except ImportError:
                pytest.fail(f"Core dependency '{dep}' not importable")

    @pytest.mark.compatibility
    def test_zeroeval_imports(self):
        """Test that all main ZeroEval modules can be imported."""
        zeroeval_modules = [
            "zeroeval",
            "zeroeval.core",
            "zeroeval.core.dataset_class",
            "zeroeval.core.experiment_class",
            "zeroeval.core.evaluator_class",
            "zeroeval.observability",
            "zeroeval.observability.tracer",
            "zeroeval.observability.decorators",
            "zeroeval.cli",
        ]

        for module_name in zeroeval_modules:
            try:
                module = import_module(module_name)
                assert isinstance(module, ModuleType)
            except ImportError as e:
                pytest.fail(f"ZeroEval module '{module_name}' not importable: {e}")

    @pytest.mark.compatibility
    def test_openai_compatibility(self):
        """Test OpenAI library compatibility."""
        try:
            import openai

            # Test that we can access the expected API
            assert hasattr(openai, "OpenAI")
            assert hasattr(openai, "__version__")

            # Check version format
            version = openai.__version__
            assert isinstance(version, str)
            assert len(version) > 0

        except ImportError:
            pytest.fail("OpenAI library not available")

    @pytest.mark.compatibility
    def test_requests_compatibility(self):
        """Test requests library compatibility."""
        try:
            import requests

            # Test basic functionality
            assert hasattr(requests, "get")
            assert hasattr(requests, "post")
            assert hasattr(requests, "__version__")

            # Check version format
            version = requests.__version__
            assert isinstance(version, str)
            assert len(version) > 0

        except ImportError:
            pytest.fail("Requests library not available")


class TestFeatureCompatibility:
    """Test compatibility of specific Python features used by the SDK."""

    @pytest.mark.compatibility
    def test_type_hints_support(self):
        """Test that type hints work correctly."""
        from typing import Dict, List, Optional, Union

        # Test that we can use type hints
        def test_func(data: Dict[str, Union[str, int]]) -> Optional[List[str]]:
            return list(data.keys()) if data else None

        result = test_func({"key": "value"})
        assert result == ["key"]

    @pytest.mark.compatibility
    def test_async_support(self):
        """Test basic async/await support."""
        import asyncio

        async def test_async():
            return "async_result"

        # Test that we can run async functions
        result = asyncio.run(test_async())
        assert result == "async_result"

    @pytest.mark.compatibility
    def test_pathlib_support(self):
        """Test pathlib support."""
        from pathlib import Path

        # Test basic pathlib operations
        path = Path("test") / "file.txt"
        assert str(path) in [
            "test/file.txt",
            "test\\file.txt",
        ]  # Handle Windows/Unix paths
        assert path.name == "file.txt"
        assert path.suffix == ".txt"

    @pytest.mark.compatibility
    def test_context_manager_support(self):
        """Test context manager support."""

        class TestContextManager:
            def __enter__(self):
                return "context_value"

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        with TestContextManager() as value:
            assert value == "context_value"

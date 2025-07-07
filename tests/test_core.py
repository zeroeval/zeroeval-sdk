"""Core SDK functionality tests."""

import pytest

from zeroeval import __version__
from zeroeval.core.dataset_class import Dataset
from zeroeval.core.experiment_class import Experiment
from zeroeval.observability.tracer import Tracer


class TestSDKBasics:
    """Test basic SDK functionality."""

    @pytest.mark.core
    def test_version_is_defined(self):
        """Test that the SDK version is properly defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    @pytest.mark.core
    def test_dataset_creation(self):
        """Test that Dataset can be created and configured."""
        dataset = Dataset(
            name="test_dataset",
            data=[
                {"id": 1, "input": "What is 2+2?", "expected": "4"},
                {"id": 2, "input": "What is 3+3?", "expected": "6"},
            ],
        )
        assert dataset.name == "test_dataset"
        assert len(dataset.data) == 2
        assert dataset.data[0]["input"] == "What is 2+2?"

    @pytest.mark.core
    def test_experiment_creation(self):
        """Test that Experiment can be created and configured."""
        # Create a dataset and a simple task function
        dataset = Dataset(
            name="test_dataset",
            data=[{"input": "test", "expected": "result"}],
        )

        def simple_task(row):
            return f"processed: {row['input']}"

        experiment = Experiment(
            dataset=dataset,
            task=simple_task,
            name="test_experiment",
        )
        assert experiment.name == "test_experiment"
        assert experiment.dataset.name == "test_dataset"
        assert experiment.task == simple_task

    @pytest.mark.core
    def test_tracer_singleton(self):
        """Test that Tracer is a singleton."""
        tracer1 = Tracer()
        tracer2 = Tracer()
        assert tracer1 is tracer2


class TestSDKIntegration:
    """Test SDK component integration."""

    @pytest.mark.core
    def test_dataset_with_experiment(self):
        """Test that Dataset and Experiment work together."""
        dataset = Dataset(
            name="integration_test",
            data=[{"id": 1, "input": "test", "expected": "result"}],
        )

        def task_function(row):
            return row["input"].upper()

        experiment = Experiment(
            dataset=dataset,
            task=task_function,
            name="integration_experiment",
        )

        # Verify they can be used together
        assert experiment.dataset.name == dataset.name
        assert len(dataset.data) == 1
        assert experiment.task == task_function

    @pytest.mark.core
    def test_tracer_with_mock_writer(self, tracer, mock_span_writer):
        """Test that tracer works with mock writer."""
        # The tracer fixture already has a mock writer
        assert hasattr(tracer, "_writer")
        assert len(tracer._writer.spans) == 0

        # Test that we can access the standalone mock writer too
        assert len(mock_span_writer.spans) == 0
        mock_span_writer.write([{"test": "span"}])
        assert len(mock_span_writer.spans) == 1

    @pytest.mark.core
    def test_top_level_imports(self):
        """Test that key functionality can be imported from top level."""
        from zeroeval import Dataset, Experiment, set_tag, span

        # Test that we can create instances
        dataset = Dataset(name="test", data=[])

        def dummy_task(row):
            return "test"

        experiment = Experiment(dataset=dataset, task=dummy_task, name="test")

        assert dataset.name == "test"
        assert experiment.name == "test"
        assert callable(span)
        assert callable(set_tag)

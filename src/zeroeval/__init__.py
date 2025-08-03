"""
ZeroEval SDK main initialization.
Exposes core functionality for experiment decoration and running.
"""
from .core.dataset_class import Dataset
from .core.task import task
from .core.evaluation import evaluation
from .core.metrics import column_metric, run_metric
from .core.init import init
from .providers import ZeroEvalOTLPProvider, SingleProcessorProvider

__all__ = ["Dataset", "task", "evaluation", "column_metric", "run_metric", "init", "ZeroEvalOTLPProvider", "SingleProcessorProvider", "langfuse_zeroeval"]

# Optional: Add version info
__version__ = "0.1.0"
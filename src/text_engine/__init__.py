"""Public text-engine entry points, loaded only when requested.

Importing the package does not download NLTK data or initialize model/plotting
libraries. Optional helpers load their dependencies when called.
"""

from importlib import import_module
from .theme_config import THEME_KEYWORDS

__all__ = ["TextAnalysisEngine", "THEME_KEYWORDS", "run_eda", "compare_theme_models"]


def __getattr__(name):
    modules = {"TextAnalysisEngine": ".engine", "run_eda": ".plot",
               "compare_theme_models": ".model_comparison"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

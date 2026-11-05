"""Public text-engine entry points, loaded only when requested.

Importing the package does not download NLTK data or initialize model/plotting
libraries. Optional helpers load their dependencies when called.
"""

from importlib import import_module
from .theme_config import THEME_KEYWORDS

# Public names exposed by this package, including lazily resolved functions.
__all__ = ["TextAnalysisEngine", "THEME_KEYWORDS", "run_eda", "compare_theme_models",
           "classify_zero_shot"]


def __getattr__(name):
    """Resolve a public API on first access without loading every analysis module.

    Python calls this only when a requested attribute is not already present.
    Caching the resolved value in globals() makes subsequent access ordinary
    attribute lookup. Unknown names raise AttributeError for normal Python behavior.
    """
    # Relative module paths are resolved against this package. import_module can
    # also load the requested zero-shot.py filename despite its hyphen.
    modules = {"TextAnalysisEngine": ".engine", "run_eda": ".plot",
               "compare_theme_models": ".model_comparison",
               "classify_zero_shot": ".zero-shot"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    # Avoid repeatedly importing/looking up the same public function or class.
    globals()[name] = value
    return value

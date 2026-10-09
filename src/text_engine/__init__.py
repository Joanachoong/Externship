"""Expose the text analysis package's public entry points.

Import ``TextAnalysisEngine`` for cleaning and theme summaries,
``THEME_KEYWORDS`` for the default workplace theme vocabulary, and ``run_eda``
for printed summaries and exploratory charts of Glassdoor review columns.
Importing this package also initializes the NLTK resources in ``clean`` and
the Seaborn plotting style in ``analyst_text``.
"""

from .engine import TextAnalysisEngine
from .theme_config import THEME_KEYWORDS
from .analyst_text import run_eda

__all__ = ["TextAnalysisEngine", "THEME_KEYWORDS", "run_eda"]

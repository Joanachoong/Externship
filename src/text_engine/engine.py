"""Coordinate text cleaning and keyword-based analysis of DataFrame columns.

``TextAnalysisEngine`` uses the default or a caller-supplied theme dictionary
to summarize non-null text entries with document counts, theme counts, and
the 20 most frequent cleaned words. It also compares theme mentions in review
pros and cons as counts and percentages of each column's non-null entries.
Theme labels indicate keyword matches, not sentiment predictions.
"""

from collections import Counter
import pandas as pd

from .clean import clean
from .theme import extract_themes
from .theme_config import THEME_KEYWORDS

class TextAnalysisEngine:
    """Reuse a theme vocabulary and minimum keyword-hit threshold for analysis."""

    def __init__(self, theme_dict: dict = None, min_hits: int = 1):
        """Use default themes when ``theme_dict`` is omitted or empty.

        ``min_hits`` is the minimum number of matching dictionary entries
        required to assign a theme to one text entry.
        """
        self.theme_dict = theme_dict or THEME_KEYWORDS
        # The engine stores configuration only; initialization does not train or
        # load a prediction model. Matching is delegated to the keyword helper.
        self.min_hits = min_hits

    def clean(self, text: str) -> str:
        """Normalize a string with the shared English preprocessing pipeline."""
        return clean(text)

    def extract_themes(self, text: str) -> list:
        """Return theme names matching this engine's vocabulary and threshold."""
        return extract_themes(text, self.theme_dict, self.min_hits)

    def analyze_column(self, df: pd.DataFrame, column_name: str) -> dict:
        """
        Summarize an existing text column, dropping nulls and converting to strings.

        Return ``n_docs`` (non-null entry count), ``theme_counts`` (a Counter
        of entries mentioning each theme), and ``top_words`` (up to 20
        cleaned word/count pairs). An entry can contribute to multiple themes
        but contributes at most once to each theme.
        """
        # Preserve the lexical baseline's denominator: nulls are excluded but
        # non-null empty strings still count as documents (unlike semantic input).
        texts = df[column_name].dropna().astype(str)
        all_themes = []
        all_words = []

        for text in texts:
            themes = self.extract_themes(text)
            # Theme labels are unique per field, while repeated words contribute
            # repeated tokens. These counters therefore measure different units.
            all_themes.extend(themes)
            all_words.extend(self.clean(text).split())

        return {
            "n_docs": len(texts),
            "theme_counts": Counter(all_themes),
            "top_words": Counter(all_words).most_common(20),
        }

    def compare_pros_cons(self, df: pd.DataFrame,
                          pros_col: str = "review_pros",
                          cons_col: str = "review_cons") -> pd.DataFrame:
        """
        Compare theme mentions in two existing review columns.

        Return a DataFrame with theme names, counts, and percentages for pros
        and cons, sorted by descending cons percentage. Each percentage uses
        its own column's non-null entry count and is rounded to one decimal.
        A column with no non-null entries contributes zero percentages. If
        neither column matches a theme, return an empty table with the same schema.
        """
        pros = self.analyze_column(df, pros_col)
        cons = self.analyze_column(df, cons_col)

        # Include themes seen on either side; a missing Counter entry is zero.
        themes = sorted(set(pros["theme_counts"]) | set(cons["theme_counts"]))
        rows = []
        for theme in themes:
            p_count = pros["theme_counts"].get(theme, 0)
            c_count = cons["theme_counts"].get(theme, 0)
            # Each column has its own denominator. An entry can match multiple
            # themes, so the percentages across themes need not sum to 100%.
            rows.append({
                "theme": theme,
                "pros_count": p_count,
                "pros_pct": round(100 * p_count / pros["n_docs"], 1) if pros["n_docs"] else 0,
                "cons_count": c_count,
                "cons_pct": round(100 * c_count / cons["n_docs"], 1) if cons["n_docs"] else 0,
            })
        # Explicit columns keep sorting safe even when neither side matches any
        # theme. Pros/cons polarity comes from the columns, not a sentiment model.
        return pd.DataFrame(rows, columns=["theme", "pros_count", "pros_pct",
                                           "cons_count", "cons_pct"]).sort_values("cons_pct", ascending=False)

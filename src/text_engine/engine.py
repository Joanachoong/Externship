from collections import Counter
import pandas as pd

from .clean import clean
from .theme import extract_themes
from .theme_config import THEME_KEYWORDS

class TextAnalysisEngine:
    def __init__(self, theme_dict: dict = None, min_hits: int = 1):
        self.theme_dict = theme_dict or THEME_KEYWORDS
        self.min_hits = min_hits

    def clean(self, text: str) -> str:
        return clean(text)

    def extract_themes(self, text: str) -> list:
        return extract_themes(text, self.theme_dict, self.min_hits)

    def analyze_column(self, df: pd.DataFrame, column_name: str) -> dict:
        """
        Returns:
          - theme_counts
          - top_words
          - n_docs
        """
        texts = df[column_name].dropna().astype(str)
        all_themes = []
        all_words = []

        for text in texts:
            themes = self.extract_themes(text)
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
        Side-by-side % of reviews mentioning each theme.
        """
        pros = self.analyze_column(df, pros_col)
        cons = self.analyze_column(df, cons_col)

        themes = sorted(set(pros["theme_counts"]) | set(cons["theme_counts"]))
        rows = []
        for theme in themes:
            p_count = pros["theme_counts"].get(theme, 0)
            c_count = cons["theme_counts"].get(theme, 0)
            rows.append({
                "theme": theme,
                "pros_count": p_count,
                "pros_pct": round(100 * p_count / pros["n_docs"], 1) if pros["n_docs"] else 0,
                "cons_count": c_count,
                "cons_pct": round(100 * c_count / cons["n_docs"], 1) if cons["n_docs"] else 0,
            })
        return pd.DataFrame(rows).sort_values("cons_pct", ascending=False)
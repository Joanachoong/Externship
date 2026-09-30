"""
This folder wull include will label all dataset based on the theme 
"""

# Theme keyword dictionary – easy to edit without touching engine logic

from .clean import clean

def extract_themes(text: str, theme_dict: dict, min_hits: int = 1) -> list:
    """
    Return list of themes found in the text.
    min_hits = 1 → permissive; raise to 2 for stricter matching.
    """
    cleaned = clean(text)
    if not cleaned:
        return []

    found = []
    for theme, keywords in theme_dict.items():
        hits = sum(1 for kw in keywords if kw in cleaned)
        if hits >= min_hits:
            found.append(theme)
    return found
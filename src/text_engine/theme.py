"""Assign workplace themes by searching cleaned text for configured keywords.

``extract_themes`` cleans each input string and counts matching keyword
entries per theme. Matching uses substring checks against the cleaned text;
keywords are cleaned with the same pipeline before matching.
Matching remains substring-based, so short keywords can match inside longer words. Each qualifying theme appears
once in the returned list, in dictionary order.
"""

from .clean import clean

def extract_themes(text: str, theme_dict: dict, min_hits: int = 1) -> list:
    """
    Return theme names with at least ``min_hits`` matching distinct normalized keywords.

    ``text`` must be a string and ``theme_dict`` maps theme names to keyword
    sequences. Repeated occurrences of a keyword do not add extra hits.
    The default threshold of one is permissive; two requires two matching
    entries. Return an empty list if cleaning removes all input tokens.
    """
    cleaned = clean(text)
    if not cleaned:
        return []

    found = []
    for theme, keywords in theme_dict.items():
        normalized = {clean(kw) for kw in keywords}
        hits = sum(1 for kw in normalized if kw and kw in cleaned)
        if hits >= min_hits:
            found.append(theme)
    return found

"""Print summaries and display exploratory charts for Glassdoor review text.

The four target DataFrame columns are:
  1. review_cons        – worker challenges in Amazon
  2. review_pros        – what workers like about Amazon
  3. summary            – overall feedback snapshot
  4. advice_to_management – advice to management (backbone of solution proposals)

Each section produces:
  • Top-N word frequency bar chart  (cleaned, stopwords removed; default 30)
  • Combined top-N bigram frequency bar chart across all target columns
  • Theme distribution bar chart      (using THEME_KEYWORDS from theme_config)
  • Text-length histogram             (character count distribution)

Call ``run_eda(df)`` from glassdoor.ipynb to execute all four analyses.
Missing columns are skipped. Chart helpers drop nulls and convert values to
strings; text lengths use original strings, while word counts use cleaned
tokens. Theme counts use the shared keyword matcher and may overlap across
themes. Results are printed or shown with Matplotlib, rather than returned
or saved. Importing this module sets the Seaborn plotting style globally.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import pandas as pd

from .clean import clean
from .theme_config import THEME_KEYWORDS
from .theme import extract_themes

# ─────────────────────────────────────────────
# Styling constants
# ─────────────────────────────────────────────
PALETTE = {
    "review_cons":    "#e05c5c",   # red   – challenges
    "review_pros":    "#4caf81",   # green – positives
    "summary":        "#5b8dd9",   # blue  – overview
    "advice_to_management":  "#f5a623",   # amber – solutions
}

COLUMN_LABELS = {
    "review_cons":   "Review Cons  —  Worker Challenges",
    "review_pros":   "Review Pros  —  What Workers Like",
    "summary":       "Summary  —  Overall Feedback",
    "advice_to_management": "Advice to Management  —  Solution Backbone",
}

TARGET_COLUMNS = ["review_cons", "review_pros", "summary", "advice_to_management"]

# This changes plotting defaults for the current Python process/notebook.
sns.set_theme(style="whitegrid", font_scale=1.05)


# ─────────────────────────────────────────────
# Helper: Top-N word frequency bar chart
# ─────────────────────────────────────────────
def plot_word_frequency(texts: pd.Series, column: str, top_n) -> None:
    """
    Clean text, remove stopwords, and display the top_n most frequent tokens.

    Counts are token occurrences, not numbers of reviews mentioning a word.
    Repeated words in a long entry contribute repeatedly. Returns None and does
    not write an image file; entries that clean to nothing contribute no tokens.
    """
    color = PALETTE.get(column, "steelblue")
    label = COLUMN_LABELS.get(column, column)

    # This is the lexical/stemmed view of the corpus, separate from the original
    # strings used for semantic embeddings and length measurements.
    all_words = " ".join(texts.dropna().astype(str).apply(clean)).split()
    word_freq  = Counter(all_words)
    common     = word_freq.most_common(top_n)

    if not common:
        print(f"[WARN] No words found for column: {column}")
        return

    words  = [w for w, _ in common]
    counts = [c for _, c in common]

    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(words, counts, color=color, edgecolor="black", alpha=0.88)

    # Write exact counts above bars so small differences are visible without
    # estimating values from the y-axis.
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.4,
            str(int(yval)),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(f"Top {top_n} Most Common Words\n{label}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Helper: Top-N bigram frequency bar chart
# ─────────────────────────────────────────────
def plot_bigram_frequency(texts: pd.Series, column: str, top_n) -> None:
    """Clean text and display the top_n most frequent two-word phrases."""
    color = PALETTE.get(column, "steelblue")
    label = COLUMN_LABELS.get(column, column)

    bigrams = []
    for text in texts.dropna().astype(str):
        words = clean(text).split()
        bigrams.extend(zip(words, words[1:]))

    _plot_bigrams(Counter(bigrams).most_common(top_n), top_n, color, label)


def plot_combined_bigram_frequency(
    df: pd.DataFrame, top_n: int = 30, columns: list[str] | None = None
) -> None:
    """Display the top bigrams combined across the available target columns."""
    columns = columns or TARGET_COLUMNS
    bigrams = []
    available_columns = [column for column in columns if column in df.columns]
    for column in available_columns:
        for text in df[column].dropna().astype(str):
            words = clean(text).split()
            bigrams.extend(zip(words, words[1:]))

    common = Counter(bigrams).most_common(top_n)
    if not common:
        print("[WARN] No bigrams found across the target columns")
        return

    _plot_bigrams(
        common,
        top_n,
        "steelblue",
        "All Glassdoor Review Columns",
    )


def _plot_bigrams(common, top_n: int, color: str, label: str) -> None:
    phrases = [" ".join(words) for words, _ in common]
    counts = [count for _, count in common]

    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(phrases, counts, color=color, edgecolor="black", alpha=0.88)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.4,
            str(int(yval)),
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(f"Top {top_n} Most Common Bigrams\n{label}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Bigrams", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Helper: Theme distribution bar chart
# ─────────────────────────────────────────────
def plot_theme_distribution(texts: pd.Series, column: str) -> None:
    """
    Counts how many reviews mention each theme from THEME_KEYWORDS,
    then plots a horizontal bar chart so theme names stay readable.

    One non-null field counts at most once per theme; a field can match several
    themes. Percentages use this column's non-null count, including blank strings.
    These are keyword mentions, not discovered topics or measured sentiment.
    """
    color = PALETTE.get(column, "steelblue")
    label = COLUMN_LABELS.get(column, column)

    theme_counter: Counter = Counter()
    n_docs = 0
    for text in texts.dropna().astype(str):
        themes = extract_themes(text, THEME_KEYWORDS, min_hits=1)
        # extract_themes returns each qualifying theme only once per field.
        theme_counter.update(themes)
        n_docs += 1

    if not theme_counter:
        print(f"[WARN] No themes detected for column: {column}")
        return

    sorted_themes = sorted(theme_counter.items(), key=lambda x: x[1], reverse=True)
    # Human-readable labels are for display; the underlying IDs are unchanged.
    theme_names   = [t.replace("_", " ").title() for t, _ in sorted_themes]
    theme_counts  = [c for _, c in sorted_themes]
    theme_pcts    = [round(100 * c / n_docs, 1) if n_docs else 0 for c in theme_counts]

    fig, ax = plt.subplots(figsize=(10, max(4, len(theme_names) * 0.6)))
    bars = ax.barh(theme_names, theme_counts, color=color, edgecolor="black", alpha=0.88)

    for bar, pct in zip(bars, theme_pcts):
        width = bar.get_width()
        ax.text(
            width + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{int(width)}  ({pct}%)",
            va="center", ha="left", fontsize=10, fontweight="bold"
        )

    ax.set_title(f"Theme Distribution  (n={n_docs} reviews)\n{label}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Review Count Mentioning Theme", fontsize=12)
    # barh places its first category at the bottom; invert to show largest first.
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.65)
    ax.set_xlim(0, max(theme_counts) * 1.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Helper: Text-length histogram
# ─────────────────────────────────────────────
def plot_text_length(texts: pd.Series, column: str) -> None:
    """
    Plots a histogram of character-length distribution per review text,
    helping identify very short (low-signal) or very long (detailed) entries.

    Length is measured before cleaning and is not the tokenizer length used to
    decide model chunk sizes. Null entries are excluded; blank strings have length 0.
    """
    color = PALETTE.get(column, "steelblue")
    label = COLUMN_LABELS.get(column, column)

    lengths = texts.dropna().astype(str).str.len()
    if lengths.empty:
        print(f"[WARN] No text data found for column: {column}")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=30, color=color, edgecolor="black", alpha=0.82)

    median_len = lengths.median()
    # A median marker provides context when a few long reviews skew the histogram.
    ax.axvline(median_len, color="black", linestyle="--", linewidth=1.5,
               label=f"Median: {int(median_len)} chars")

    ax.set_title(f"Text Length Distribution\n{label}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Character Count per Review", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Helper: Print quick text stats summary
# ─────────────────────────────────────────────
def print_column_summary(df: pd.DataFrame, column: str) -> None:
    """Print row count, non-null coverage, missing values, and mean raw length.

    'Missing' means null, not an empty string. Mean length can be NaN when every
    entry is null. No analysis table is returned and no input values are modified.
    """
    label  = COLUMN_LABELS.get(column, column)
    series = df[column]
    total  = len(series)
    filled = series.notna().sum()
    empty  = total - filled
    avg_len = series.dropna().astype(str).str.len().mean()

    print(f"\n{'─'*55}")
    print(f"  📋  {label}")
    print(f"{'─'*55}")
    print(f"  Total rows   : {total}")
    print(f"  Non-null     : {filled}  ({round(100*filled/total,1) if total else 0}%)")
    print(f"  Missing      : {empty}")
    print(f"  Avg length   : {round(avg_len, 1)} characters")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────
# Main entry point — call from glassdoor.ipynb
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame, top_n: int = 30) -> None:
    """
    Runs the full EDA pipeline for all four text columns.

    Usage in glassdoor.ipynb:
    ─────────────────────────
        import sys
        sys.path.append("..")
        from text_engine.plot import run_eda
        run_eda(df)

    Parameters
    ----------
    df    : Glassdoor DataFrame with original text fields; helpers clean as needed
    top_n : Number of top words to show per column (default 30)

    Processes the columns in TARGET_COLUMNS order. Missing columns are skipped,
    allowing partial datasets. Results are printed/shown, not returned or saved.
    """
    available_columns = [col for col in TARGET_COLUMNS if col in df.columns]
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            print(f"[SKIP] Column '{col}' not found in DataFrame.")
            continue

        texts = df[col]
        print_column_summary(df, col)
        print(f"  → Word Frequency Chart  ({col})")
        plot_word_frequency(texts, col, top_n=top_n)

    print("  → Combined Bigram Frequency Chart (all four columns)")
    plot_combined_bigram_frequency(df, top_n=top_n, columns=available_columns)

    for col in available_columns:
        texts = df[col]
        print(f"  → Theme Distribution Chart  ({col})")
        plot_theme_distribution(texts, col)

        print(f"  → Text Length Histogram  ({col})")
        plot_text_length(texts, col)

        print("\n")


def plot_theme_model_comparison(result):
    """Display per-column theme percentages from a semantic comparison result.

    BERTopic labels are mapped to the predefined taxonomy; the title explicitly
    identifies provisional mappings. Discovered novel topics remain in result.

    Expects compare_theme_models()/apply_topic_mapping() output. The fixed 2x2
    layout accommodates the four standard text columns. Shows and returns the
    figure without saving it. Percentages describe each column's nonblank fields.
    """
    distributions = result['theme_distribution']
    columns = list(result['coverage']['column'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), squeeze=False)
    for ax, column in zip(axes.flat, columns):
        data = distributions[distributions['column'] == column]
        # Convert one row per (theme, method) into side-by-side percentage bars.
        # Only mapped predefined themes appear here; raw novel topics need their
        # own inspection through result['topics'] and result['predictions'].
        table = data.pivot(index='theme', columns='method', values='pct')
        table.plot.barh(ax=ax, color=['#f5a623', '#5b8dd9'])
        ax.set_title(COLUMN_LABELS.get(column, column))
        ax.set_xlabel('Percentage of nonblank text entries')
        ax.set_ylabel('Theme')
        ax.legend(title='Method')
    # Keep the mapping status visible so provisional suggestions are not mistaken
    # for human-approved meanings. This comparison does not demonstrate accuracy.
    status = result['metadata']['mapping_status'].replace('_', ' ')
    fig.suptitle(f'Zero-shot and BERTopic — {status}', fontsize=15)
    plt.tight_layout()
    plt.show()
    return fig

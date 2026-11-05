# Text Analysis Engine

## Purpose

This package analyzes English workplace review text, particularly Amazon
Glassdoor reviews. It helps identify employee challenges, positive experiences,
and suggestions for management by cleaning text, counting common words, and
grouping reviews into predefined workplace themes.

The main analysis uses a **rule-based keyword matching algorithm**. Theme labels
come from a manually defined dictionary; this package does not train a classifier
or predict positive and negative sentiment. Comparing the `review_pros` and
`review_cons` columns uses the reviewer's existing separation of pros and cons.

The exploratory analysis supports these DataFrame columns:

| Column | Purpose |
| --- | --- |
| `review_cons` | Explore employee difficulties and negative experiences. |
| `review_pros` | Explore what employees appreciate. |
| `summary` | Examine the overall review summary. |
| `advice_to_management` | Explore employee suggestions for improvement. |

## Data cleaning methods

The `clean(text)` function in `clean.py` applies these steps **in this order**:

| Step | Method used | What it does |
| --- | --- | --- |
| 1 | Lowercasing with `str.lower()` | Makes uppercase and lowercase words consistent. |
| 2 | ASCII punctuation removal with `str.translate()` and `string.punctuation` | Deletes punctuation characters before tokenization. |
| 3 | Whitespace tokenization with `str.split()` | Splits the text into individual tokens. |
| 4 | NLTK English stopword removal | Removes common words such as `the`, `and`, and `is`. |
| 5 | NLTK `WordNetLemmatizer` | Converts tokens to dictionary base forms where possible, such as `cars` to `car`. |
| 6 | NLTK English `SnowballStemmer` | Applies stemming rules to reduce words to stems, such as `working` to `work`. |
| 7 | Joining with spaces | Returns the processed tokens as one string. |

Lemmatization runs before stemming. The lemmatizer receives no part-of-speech
argument, so it uses its default noun interpretation. The resulting text may
contain stems that are not complete dictionary words.

The DataFrame analysis and chart functions drop null entries and convert remaining
values to strings before processing. Calling `clean()` directly requires a string.
These methods do not remove duplicate reviews, correct spelling, or filter reviews
by language.

## Algorithms and analysis techniques

### Rule-based theme matching

`extract_themes()` in `theme.py` implements dictionary-based, multi-label theme
assignment:

1. Clean the input review with `clean()`.
2. Loop through each theme and its keyword list in `THEME_KEYWORDS`.
3. Clean each keyword with the same pipeline, then check for a substring match.
4. Count distinct matching normalized keywords for that theme.

The matching rule is:

A review can receive multiple themes. Each theme is counted once per review;
repeating a keyword in the review does not add extra hits. Allow multiple theme in one column (no restriction)

The default themes are `physical_demand`, `management`, `workplace_politics`,
`bathroom_access`, `work_pace`, `career_growth`, `pay_benefits`,
`schedule_flexibility`, and `culture_people`.
The dictionary is editable in `theme_config.py`.

### Word frequency and theme percentages

`TextAnalysisEngine.analyze_column()` uses Python's `collections.Counter` to
count cleaned word occurrences and theme mentions across non-null entries. It
returns the number of documents, theme counts, and the top 20 word/count pairs.

`compare_pros_cons()` compares themes across the pros and cons columns using:

```text
theme percentage = 100 × entries mentioning the theme / non-null entries in the column
```

Percentages are rounded to one decimal place, and results are sorted by descending
cons percentage. Because themes can overlap, their percentages can total more
than 100%.

### Part-of-speech tagging and dependency parsing

`clean.py` also provides linguistic analysis helpers that operate on **raw text**:

| Helper | Method | Output |
| --- | --- | --- |
| `pos_tag_text(text)` | NLTK `word_tokenize()` followed by pretrained `pos_tag()` | Token/tag pairs describing grammatical roles. |
| `parse_syntax(text)` | spaCy's pretrained `en_core_web_sm` pipeline | Each token's text, part of speech, dependency relation, and grammatical head. |

These helpers are separate from `clean()` and are not automatically called by
`TextAnalysisEngine` or `run_eda()`. They use pretrained NLP components; this
repository does not train them.

### Bag of Words and TF-IDF

`text_representation(corpus)` cleans each document and creates two numerical
representations using scikit-learn:

- **Bag of Words (BoW)** with `CountVectorizer`: counts vocabulary token occurrences
  in each document without preserving word order.
- **Term Frequency–Inverse Document Frequency (TF-IDF)** with `TfidfVectorizer`:
  weights token frequency by how uncommon a token is across the supplied corpus.

Both vectorizers use their default settings, including single-word features
(unigrams). Their vocabularies are fitted to the supplied corpus. The function
returns `bow_df`, `tfidf_df`, and the two fitted vectorizers. The TF-IDF DataFrame
is rounded to three decimal places.

These are feature representation methods. The current engine does not use their
matrices to train a prediction model or assign themes.

## Exploratory data analysis

`run_eda(df, top_n=30)` in `plot.py` prints a summary and displays three
charts for each available target column:

- A frequency bar chart of the top cleaned words.
- A combined frequency bar chart of the top cleaned bigrams across all four columns.
- A theme distribution chart with counts and percentages.
- A histogram of original text lengths with a median marker.

The printed summary includes total rows, non-null entries, missing entries, and
average text length. Missing columns are skipped. Charts are displayed using
Matplotlib and Seaborn; the function does not save them or return an analysis table.

## Files

| File | Responsibility |
| --- | --- |
| `__init__.py` | Exports `TextAnalysisEngine`, `THEME_KEYWORDS`, and `run_eda`. |
| `clean.py` | Cleans text and provides POS tagging, dependency parsing, BoW, and TF-IDF helpers. |
| `theme.py` | Matches cleaned text against theme keywords. |
| `theme_config.py` | Defines workplace keyword themes and full-language semantic descriptions. |
| `engine.py` | Summarizes columns and compares pros and cons. |
| `plot.py` | Prints exploratory summaries and displays EDA and model-comparison charts. |
| `zero-shot.py` | Scores predefined themes using multilingual zero-shot NLI; exports `classify_zero_shot` through the package. |
| `_semantic.py` | Shares tokenizer-based chunking between zero-shot classification and BERTopic. |
| `model_comparison.py` | Calls the zero-shot classifier and runs BERTopic discovery, alignment, and evaluation. |


## Current limitations

- Keywords and review text now share the same cleaning pipeline. Distinct keywords
  that collapse to the same stem count as one hit; meaning and phrase boundaries
  can still be lost during normalization.
- Matching uses substrings rather than whole words. For example, `pay` can match
  inside `payment`, potentially assigning a theme too broadly.
- Stopword removal can discard negation, and stemming can remove useful nuance.
  Theme matching does not account for context, sarcasm, or sentiment.
- Punctuation is deleted rather than replaced with spaces, so a hyphenated phrase
  such as `fast-paced` becomes `fastpaced`.
- `compare_pros_cons()` returns an empty table with named columns when neither
  column matches a theme. Non-null empty strings can still be counted as documents
  by lexical summaries; the semantic comparison excludes blank strings.
- BoW and TF-IDF can fail with an empty vocabulary. The helper converts sparse
  matrices to dense DataFrames, which can use substantial memory for large corpora.

These limitations should be considered when interpreting results. The outputs
describe keyword mentions and word usage; they do not establish causes or measure
the accuracy of a trained classifier.


## Multi-label zero-shot and BERTopic comparison

Zero-shot classification is implemented separately in `zero-shot.py`. The
comparison still runs both methods. To score texts independently of BERTopic:

```python
from text_engine import classify_zero_shot

zero_shot = classify_zero_shot(["Good pay, but unpredictable shifts."])
scores = zero_shot["scores"]  # One {theme_id: score} dictionary per input text.
```

The package handles importing the hyphenated filename. Model loading is deferred
until classification is called; the result also reports loading and inference times.

The final comparison call is in `src/notebook/glassdoor.ipynb`. Its setup locates
this repository from either the repository root or the notebook directory. Install
`requirements.txt` into the notebook kernel's Python environment. A prepared local
environment is available at `.venv-theme/bin/python`; alternatively run:

```sh
.venv-theme/bin/python scripts/run_theme_comparison.py
```

`model_comparison.py` compares two different approaches on the same nonblank
review fields across all four columns:

- **Zero-shot NLI:** the multilingual pretrained
  `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` model independently scores every
  description in `THEME_DESCRIPTIONS`, using `multi_label=True`.
- **BERTopic:** multilingual MiniLM sentence embeddings, UMAP dimension reduction,
  HDBSCAN leaf clustering and c-TF-IDF topic representations discover themes.
  Leaf selection helps avoid a single broad cluster on this small corpus; the
  configurable alternative is `cluster_selection_method="eom"`. English stopwords
  and frequent-word reduction apply only to topic keyword representations,
  preserving original text for multilingual semantic embeddings. The number
  of topics is inferred from the data, with no fixed requested topic count.
  BERTopic's approximate topic distributions support multiple topics per text;
  the primary hard cluster is retained only as an additional diagnostic.

Every label whose score meets its configurable threshold is retained. There is
**no top-k cap**, no forced single label, and no forced label for an unmatched text.
The predefined zero-shot taxonomy is extensible; topic discovery can reveal themes
outside it. An unlimited label count still requires candidate descriptions for
zero-shot classification.

Semantic models use `prepare_semantic_text()`, which only normalizes whitespace
and preserves grammar, negation, punctuation, and case. Stemming and stopword
removal remain available through `clean()` for lexical summaries. Model-tokenizer
chunks cover long texts; zero-shot and approximate topic scores use maximum chunk
scores, while document embeddings average normalized chunk embeddings. Longer
texts may consequently cross more thresholds. Topic fitting pools all four fields,
so columns with more or longer entries influence discovered topic representations.

Outputs include complete scores, per-field labels, discovered topics, provisional
topic alignment, per-column theme distributions, unlabeled and multi-label coverage,
Jaccard agreement, exact-set agreement, runtime and package metadata, and an input
hash. Saved local CSV/JSON files are in the ignored `.cache/theme_comparison/`
directory. JSON preserves list/dictionary fields for later analysis.

NLI scores, topic-distribution weights and topic-mapping cosine similarities are
not interchangeable or calibrated probabilities. Default thresholds (0.5, 0.1,
and 0.35 respectively) are initial settings for review and validation. Discovery
may produce no topics when everything is an outlier, which is reported rather
than replaced with a forced theme.

BERTopic topics and predefined themes are different label spaces. Initial alignment
uses cosine similarity of topic keyword representations and theme descriptions.
This mapping is provisional. Inspect representative reviews and then call
`apply_topic_mapping()` with every discovered topic ID mapped to any number of
approved theme IDs (or an empty list for a novel topic). This updates comparison
tables without rerunning models; novel topic IDs remain in the predictions.

Agreement does not establish accuracy or a winning method. Independently annotate
held-out review fields and pass `gold` labels keyed by `document_id` to compute
micro/macro precision, recall, F1 and exact-set accuracy. Unannotated entries are
excluded. Keep fields from one review in the same validation/test split. Prefer
zero-shot when its held-out F1 is within a predeclared acceptable difference and
reviewed BERTopic discoveries offer little useful additional coverage. Runtime
figures distinguish cold model loading from scoring/fitting, which are different
workloads rather than directly equivalent throughput measurements.

References: [zero-shot multi-label classification](https://huggingface.co/docs/inference-providers/main/tasks/zero-shot-classification),
[zero-shot model card](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli),
[BERTopic topic distributions](https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html),
[multilingual sentence embedding model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

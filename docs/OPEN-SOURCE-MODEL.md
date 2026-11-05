# Open-source models and methods used for Glassdoor analysis

Documented on November 5, 2026, from the current source code and saved notebook outputs. No model was rerun to prepare this document.

The aim is to understand employee experiences in the Glassdoor reviews, including themes that were not predefined. The main discovery method is **BERTopic with multilingual sentence embeddings**. A separate zero-shot classifier provides a predefined-theme comparison, and an earlier notebook used **TextBlob for sentiment analysis**. The project does not currently use a BERT sentiment classifier or fine-tune transformer weights on Glassdoor labels.

The selection reasons below describe how each component fits the project goals and available data. They do not imply that a benchmark proved it was the best available model.

## Evidence and dataset scope

The primary reference is [glassdoor.ipynb](../src/notebook/glassdoor.ipynb), especially the cells containing `comparison_result = compare_theme_models(...)`, `display(comparison_result["topics"])`, and `REVIEWED_TOPIC_MAPPING`. The latest saved comparison uses **nine predefined themes and 34 discovered topics**. The older [theme categorization report](../src/notebook/theme-categorization-result.md) describes a different run with seven predefined themes and 33 topics; its figures should not be substituted for the current results.

The raw [Glassdoor CSV](../data/raw/glass_reviews.csv) contains **127 reviews**. The semantic pipeline processes each nonblank text field separately:

| Field | Nonblank fields | Purpose |
| --- | ---: | --- |
| `review_cons` | 127 | Difficulties and criticisms |
| `review_pros` | 127 | Positive experiences |
| `summary` | 126 | Short overall descriptions |
| `advice_to_management` | 37 | Suggestions and requests for improvement |
| **Total** | **417** | Fields passed to the theme models |

These are 417 fields from 127 reviews, not 417 independent reviewers. Advice is available for only 37 reviewers, so its percentages describe that subset. Fields can receive multiple labels; percentages across themes need not add to 100%.

## Model and method inventory

| Component | Exact model or implementation | Role and status |
| --- | --- | --- |
| Sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Used in the current theme pipeline to represent text meaning and suggest topic-to-theme mappings. |
| Token chunking | Custom `_chunks()` in [`_semantic.py`](../src/text_engine/_semantic.py), using each model's tokenizer | Used to cover long fields. This is an algorithm, not a separate trained chunking model. |
| Topic discovery | `BERTopic` | Used to discover groups of related texts without predefined theme labels. |
| Dimension reduction | `UMAP` | Used before clustering to reduce the embedding dimensions. |
| Clustering | `HDBSCAN` | Used to discover clusters and leave some fields as outliers. |
| Topic representation | `CountVectorizer` and `ClassTfidfTransformer` | Used to describe clusters with words and two-word phrases. These are lexical representations, not additional transformer models. |
| Predefined-theme classification | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | Still executed by the saved comparison workflow, although discovery is the main research priority. Implemented separately in [`zero-shot.py`](../src/text_engine/zero-shot.py). |
| Keyword baseline | `extract_themes()` and `THEME_KEYWORDS` | Used by the lexical engine and EDA charts. A manually written dictionary, not a learned classifier. |
| Historical sentiment | TextBlob's default `PatternAnalyzer` | Used in the earlier Glassdoor sentiment notebook on `cleaned_summary`; not part of the current theme comparison. |
| Supporting language tools | NLTK English stopwords, WordNet lemmatization, and Snowball stemming | Used for lexical cleaning. NLTK POS tagging and spaCy `en_core_web_sm` are also available as optional helpers, with no measured contribution to the current theme results. |

“BERT” here refers to the transformer-based embedding approach within BERTopic. The exact embedding checkpoint is multilingual MiniLM; BERTopic is the discovery pipeline built around those embeddings, rather than a single BERT classifier trained on these reviews.

## Embedding model: multilingual MiniLM

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

The model maps sentences and paragraphs into **384-dimensional vectors**. Its model card lists 50 languages, making it suitable for an initial analysis of reviews that include English, German, and Spanish. Related wording can therefore be grouped by meaning rather than requiring the same keyword spelling. [Embedding model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

**Reason for selection:** the project has a small, unlabeled dataset, needs semantic similarity for discovery, includes multiple languages, and runs inference locally on CPU. An existing sentence encoder supplies those representations without requiring a labeled training dataset.

In [`model_comparison.py`](../src/text_engine/model_comparison.py), `SentenceTransformer` encodes the text chunks with normalized embeddings. For a field spanning multiple chunks, the code averages their vectors and normalizes the result again. BERTopic receives one vector per field. Natural wording, punctuation, and negation are preserved.

**Observed performance:** the resulting pipeline produces recognizable topic representations about lifting, promotion, scheduling, and benefits. This supports its usefulness for exploration, but it does not isolate the encoder's accuracy: clustering, representation settings, and input quality also affect the result. There is no saved embedding benchmark against alternative encoders on this dataset.

## Chunking: tokenizer-based text splitting

There is **no independently trained chunking model**. The custom [`_chunks()` helper](../src/text_engine/_semantic.py) counts tokens with the relevant model's tokenizer and returns consecutive, nonoverlapping text chunks.

| Path | Content-token budget | How chunk results are combined |
| --- | --- | --- |
| Zero-shot classification | 256 tokens per chunk | Highest score for each theme across a field's chunks |
| Sentence embeddings | `encoder.max_seq_length - 2`, with a minimum of 1 | Average chunk embeddings, then normalize the field vector |
| Approximate topic assignment | Reuses the embedding-path chunks | Highest approximate weight for each topic across chunks |

**Reason for selection:** token-based splitting follows model input lengths and covers long fields instead of silently discarding the ending. It uses the existing tokenizers without another model to train or maintain.

**Limitations and evaluation:** chunks can cross sentence boundaries and have no overlap. The final short chunk receives equal weight in embedding averaging. Maximum aggregation gives long fields more opportunities to cross a label threshold, and its resulting scores need not sum to one. Review long examples for preserved context and inspect the chunk boundaries; no separate chunking-quality metric has been recorded.

## Theme discovery: BERTopic, UMAP, HDBSCAN, and c-TF-IDF

**Reason for selection:** discover the hidden topic other than the pre-defined topic 

The current implementation pools all four text columns during fitting, then reports results separately by source column. The predefined theme descriptions do not determine the discovered clusters.

| Setting | Current value | Purpose |
| --- | --- | --- |
| UMAP dimensions | 5 | Reduce embeddings before density clustering |
| UMAP neighbors | `min(15, n_fields - 1)` | Control the neighborhood used for dimension reduction |
| UMAP distance | Cosine | Compare semantic vectors |
| HDBSCAN minimum cluster size | 5 | Allow small recurring groups; not a requested number of topics |
| HDBSCAN minimum samples | 2 | Set the density requirement separately from cluster size |
| HDBSCAN selection | `leaf` | Select finer groups; the historical report records an earlier broad configuration collapsing most fields into one topic |
| Topic terms | Unigrams and bigrams; English stopwords removed | Produce readable topic keywords |
| c-TF-IDF | Frequent-word reduction enabled | Reduce generic words in topic representations |
| Random seed | 42 | Support reproducibility of a configured run |

### Observed discovery results

The current saved run contains **34 topics**, plus **135 primary-cluster outlier fields** out of 417 (**32.4%**). Ordinary cluster sizes range from 5 to 16 fields. The 34 topics are candidate groupings requiring review, not 34 validated workplace themes.

| Topic ID in this run | Primary-cluster fields | Observed representation | Interpretation for review |
| --- | ---: | --- | --- |
| 1 | 16 | Noise, rules, repetitive tasks | Candidate concerns beyond the broad taxonomy, but representative texts also mix physical demands, safety, and workload. |
| 6 | 11 | Heavy lifting, physical strength | A recognizable physical-demand group. |
| 10 | 10 | Flexible schedule, scheduling hours | A recognizable scheduling group. |
| 12 | 8 | Politics, promotion, promotion based on results | A candidate politics/fairness group whose representatives still need checking. |
| 15 | 8 | Break time, breaks, forced work | Contains detailed break concerns, but also a Spanish representative about flexible hours and weekend supplements. This indicates mixed content. |
| 25 | 6 | Mixed German terms and team/management wording | Needs language-aware review before assigning a precise theme name. |

Topic IDs belong to this fitted run and may change after refitting. English-only keyword stopwords also leave common German or Spanish words in the representations. These examples show both useful discovery and quality limitations.

## Zero-shot classification and topic-to-theme alignment

**Model:** `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, a multilingual natural language inference model fine-tuned on MNLI/XNLI. Its distilled architecture offers a relatively compact comparison model. [NLI model card](https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli).

**Reason for selection:** it can score workplace theme descriptions without training on Glassdoor labels. It is a useful predefined-theme baseline, while BERTopic serves the broader discovery goal. Moving the implementation into `zero-shot.py` did not disable it: `compare_theme_models()` still calls it.

The model independently evaluates hypotheses such as “This text discusses work schedules and working hours.” A theme is selected at **0.5** or above. The nine current candidate IDs are `physical_demand`, `management`, `workplace_politics`, `bathroom_access`, `work_pace`, `career_growth`, `pay_benefits`, `schedule_flexibility`, and `culture_people`.

The separate topic-alignment step embeds BERTopic keywords and theme descriptions, then proposes mappings at cosine similarity **0.35** or above. It does not create new clusters. One topic can map to several predefined themes or none. The saved mapping status is `provisional_embedding_mapping`; human-approved mappings have not been supplied.

**Observed limitation:** saved prediction `1:summary`, whose text is “Good,” receives a `bathroom_access` NLI score of approximately **0.614**. That text provides no evidence for bathroom restrictions. This spot check demonstrates why a score above the cutoff cannot be treated as proof of correctness; it is not an estimate of overall error rate.

## Lexical baseline and optional language helpers

[`engine.py`](../src/text_engine/engine.py), [`theme.py`](../src/text_engine/theme.py), and [`theme_config.py`](../src/text_engine/theme_config.py) provide an interpretable keyword baseline. The active lexical engine uses uppercase `THEME_KEYWORDS`; the additional lowercase `theme_keywords` dictionary in the configuration is not automatically used by the engine.

**Reason for keeping the baseline:** explicit keywords make matches easy to inspect and provide a simple comparison with semantic methods. The matcher normalizes keywords and review text, then checks substrings with a default minimum of one matching distinct normalized keyword. It cannot establish sentiment or reliably resolve negation, and short substrings can produce false matches.

NLTK tools support lexical cleaning and optional English POS tagging. The optional spaCy `en_core_web_sm` helper exposes grammatical dependencies. Neither POS tagging nor spaCy parsing is called by the current BERTopic comparison, and no saved result establishes that either improved theme accuracy. They should be documented as supporting capabilities rather than evaluated theme models.

## How to read the current notebook scores

Use the notebook's displayed tables and the matching local exports in `.cache/theme_comparison/`. Topic counts and coverage were cross-checked between those sources for this document. Cache files are ignored by Git, so preserve exports with any shared analysis.

| Output or score | What it tells us | What it cannot establish |
| --- | --- | --- |
| `zero_shot_scores` | NLI relevance to each supplied theme description | Percentage certainty that a human will accept the label |
| `bertopic_scores` | Approximate discovered-topic weights | Classification accuracy or HDBSCAN membership probability |
| `topic_alignment.cosine_scores` | Semantic similarity between topic keywords and theme descriptions | Human-approved topic meaning |
| `topics.Count` | Fields assigned to each primary cluster | Distinct reviewers mentioning a validated theme |
| `coverage` | Unlabeled fields, multiple labels, and average labels | Whether those labels are correct |
| `theme_distribution` | Frequency of predefined labels by column/method | Sentiment, severity, or prevalence of topics outside the taxonomy |
| `agreement` | Similarity between the two methods' predefined label sets | Agreement with human judgment |
| `gold_metrics` | Precision, recall, F1, and exact-set accuracy when human labels are provided | Accuracy for novel topics outside the evaluated vocabulary |

### Coverage in the latest saved run

Each entry below is the number of fields with **no selected labels**. BERTopic discovery counts raw topic IDs; mapped output counts predefined theme IDs, so their label spaces differ.

| Field | Fields analyzed | Zero-shot unlabeled | BERTopic discovery unlabeled | BERTopic mapped unlabeled |
| --- | ---: | ---: | ---: | ---: |
| Cons | 127 | 60 | 25 | 46 |
| Pros | 127 | 36 | 29 | 34 |
| Summary | 126 | 58 | 28 | 71 |
| Advice | 37 | 10 | 11 | 15 |

Discovery assigns at least one approximate topic to 324 of 417 fields (**77.7%**). That coverage is different from the primary-cluster outlier rate. Mapping can remove raw-topic labels that have no predefined equivalent or expand a topic into several themes. Higher coverage alone is not evidence of better results.

### Agreement in the latest saved run

| Field | Mean Jaccard | Exact-set agreement | Both methods empty |
| --- | ---: | ---: | ---: |
| Cons | 0.138 | 21.3% | 27 |
| Pros | 0.255 | 13.4% | 14 |
| Summary | 0.088 | 31.0% | 39 |
| Advice | 0.130 | 18.9% | 7 |

Jaccard is shared labels divided by all distinct labels assigned by either method. Its overall mean is approximately **0.164**, calculated over **330 nonempty label unions**. The remaining 87 fields are excluded from mean Jaccard but count as exact matches when both methods return nothing. Neither figure identifies which method is right.

## Evaluation plan: manual review for a small dataset

**Current status:** the notebook sets `REVIEWED_TOPIC_MAPPING = None` and `GOLD_LABELS = {}`. The exported `gold_metrics` table is empty and metadata says `accuracy_status: not_measured`. Human review is the planned evaluation method; a completed accuracy audit is not yet recorded.

Given the dataset size, the target is to manually review all **127 reviews and their 417 available fields**. Human intuition will be grounded in the actual wording and a consistent rubric, rather than accepting a high score or plausible topic name alone.

1. **Create human reference labels from the original text.** Record all applicable themes, evidence phrases, sentiment where evaluated, and uncertainty. Read the source before model predictions where practical to reduce anchoring. Allow “no clear theme,” mixed sentiment, and themes outside the predefined vocabulary. Seek a fluent reader or checked translation for unfamiliar languages.
2. **Review every discovered topic.** Read its keywords, representative excerpts, and additional members. Decide whether it is coherent, too broad, mixed, duplicated, or low-information. Give it an evidence-based name. Inspect outliers as well: an unusual concern can matter without forming a cluster.
3. **Audit individual assignments and omissions.** Mark labels as supported, unsupported, or uncertain, and record missed themes. Pay special attention to generic summaries, long fields, negation, advice requests, multilingual text, labels near cutoffs, and disagreements. Reviewing only predicted positives cannot reveal missed themes.

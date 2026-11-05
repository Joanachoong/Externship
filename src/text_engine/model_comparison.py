"""Compare multi-label NLI classification with BERTopic theme discovery.

Scores from the two methods have different meanings. Agreement is descriptive,
not accuracy. Discovered topics need human mapping before a validated comparison.
Transformer inputs preserve wording; the stemmed cleaner is used only by the
separate lexical analysis API. Model dependencies are loaded on demand.

Workflow:
    1. Expand the selected DataFrame columns into individual text documents.
    2. Ask the zero-shot module to score the predefined theme descriptions.
    3. Embed the same documents, cluster them, and extract BERTopic keywords.
    4. Suggest how discovered topics correspond to the predefined themes.
    5. Report assignments, coverage, agreement, and optional human-label metrics.

BERTopic discovery does not use the predefined themes to form its clusters.
Alignment happens afterward so both methods can be compared in one vocabulary.
Neither method here fine-tunes transformer weights or predicts sentiment.
"""

from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from time import perf_counter

from ._semantic import _chunks
from .clean import prepare_semantic_text
from .theme_config import THEME_DESCRIPTIONS

# import_module supports the requested hyphenated filename.
_zero_shot = import_module('.zero-shot', __package__)
ZERO_SHOT_MODEL = _zero_shot.ZERO_SHOT_MODEL
# This encoder produces vectors for clustering and for later topic alignment;
# it is a different model from the NLI classifier used by the zero-shot path.
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
# Each populated field is processed independently, with its source column kept.
TEXT_COLUMNS = ('review_cons', 'review_pros', 'summary', 'advice_to_management')



def _threshold(value, name):
    """Reject cutoffs outside (0, 1], including NaN, before running a model."""
    if not 0 < value <= 1:
        raise ValueError(f'{name} must be greater than zero and at most one')


def select_labels(scores, threshold):
    """Select every label meeting the threshold, without top-k or fallback.

    ``scores`` maps theme IDs or topic IDs to numeric scores. Results preserve
    dictionary order. An empty result is valid; this does not choose a winner.
    """
    _threshold(threshold, 'threshold')
    return [label for label, score in scores.items() if score >= threshold]


def prepare_documents(df, columns=TEXT_COLUMNS):
    """Return one row per non-null, nonblank review field.

    A review with both pros and cons produces two documents, not a merged text.
    Each output row retains its column, original review ID (when available), and
    zero-based row position. For example, ``2:review_cons`` identifies the cons
    field in the third input row. These positional document IDs are stable only
    while the input row order is unchanged; persist the matching input with labels.
    """
    import pandas as pd
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f'Missing review columns: {sorted(missing)}')
    rows = []
    # iloc avoids treating a possibly duplicated/custom DataFrame index as a row ID.
    for position in range(len(df)):
        row = df.iloc[position]
        for column in columns:
            value = row[column]
            if pd.isna(value):
                continue
            # Preserve grammar and negation for transformers. Whitespace-only
            # values become empty and are excluded from semantic denominators.
            text = prepare_semantic_text(str(value))
            if text:
                rows.append({'document_id': f'{position}:{column}',
                             'row_position': position,
                             'review_id': row.get('review_id', position),
                             'column': column, 'text': text})
    # Supplying columns also preserves the schema when every field is missing.
    return pd.DataFrame(rows, columns=['document_id', 'row_position', 'review_id',
                                      'column', 'text'])


def agreement_table(zero_sets, topic_sets):
    """Compare theme sets for the same documents in the same input order.

    Jaccard = size of intersection / size of union. Two empty sets yield NaN
    rather than inflating the mean Jaccard score, but are an exact-set match.
    The difference columns identify labels to inspect during disagreement review.
    These values measure agreement between methods, not agreement with the truth.
    """
    import pandas as pd
    if len(zero_sets) != len(topic_sets):
        raise ValueError('Prediction lists must refer to the same documents')
    rows = []
    for zero, topic in zip(zero_sets, topic_sets):
        # Sets make label order and any duplicate assignments irrelevant.
        zero, topic = set(zero), set(topic)
        union = zero | topic
        rows.append({'jaccard': len(zero & topic) / len(union) if union else float('nan'),
                     'exact_match': zero == topic,
                     'both_empty': not union,
                     'zero_only': sorted(zero - topic),
                     'bertopic_only': sorted(topic - zero)})
    return pd.DataFrame(rows)


def evaluate_predictions(predictions, gold, theme_names):
    """Evaluate common reviewed documents against human labels on the same taxonomy.

    gold maps document_id to an iterable of theme identifiers. Empty lists are
    valid annotations. Missing IDs are unreviewed, never treated as negative.
    Use held-out annotations for final performance after tuning thresholds.

    Only predefined theme IDs can be evaluated here; novel BERTopic topics need
    their own review. Micro metrics pool decisions across themes, while macro
    metrics average over all configured themes with equal weight. Undefined
    precision/recall contributions are set to zero by ``zero_division=0``.
    """
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.preprocessing import MultiLabelBinarizer
    # Reject mismatched IDs/labels so annotation mistakes cannot silently alter
    # the reference set. A string such as 'management' is not a list of labels.
    known_ids = set(predictions.document_id)
    if set(gold) - known_ids:
        raise ValueError('Gold annotations contain unknown document IDs')
    if any(isinstance(labels, str) for labels in gold.values()):
        raise ValueError('Gold labels must be lists, not strings')
    unknown = {label for labels in gold.values() for label in labels} - set(theme_names)
    if unknown:
        raise ValueError(f'Unknown gold themes: {sorted(unknown)}')
    reviewed = predictions[predictions.document_id.isin(gold)]
    if reviewed.empty:
        return pd.DataFrame()
    encoder = MultiLabelBinarizer(classes=list(theme_names))
    # Encode each document as one binary column per theme, in a fixed order.
    # Missing gold IDs were filtered out; they are not treated as all-negative.
    truth = encoder.fit_transform([gold[key] for key in reviewed.document_id])
    rows = []
    for method, column in [('zero_shot', 'zero_shot_labels'),
                           ('bertopic', 'bertopic_theme_labels')]:
        predicted = encoder.transform(reviewed[column])
        for average in ['micro', 'macro']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, predicted, average=average, zero_division=0)
            # Multilabel accuracy requires the entire predicted set to match.
            # It is repeated on micro/macro rows because it has no averaging mode.
            rows.append({'method': method, 'average': average,
                         'precision': precision, 'recall': recall, 'f1': f1,
                         'exact_set_accuracy': accuracy_score(truth, predicted),
                         'n_reviewed': len(reviewed)})
    return pd.DataFrame(rows)


def compare_theme_models(df, *, columns=TEXT_COLUMNS, themes=None,
                         zero_threshold=0.5, topic_threshold=0.1,
                         mapping_threshold=0.35, topic_mapping=None, gold=None,
                         zero_model=ZERO_SHOT_MODEL, embedding_model=EMBEDDING_MODEL,
                         batch_size=8, min_topic_size=5, cluster_selection_method="leaf",
                         seed=42, device=-1, embedding_device="cpu"):
    """Run both methods on the same review fields and return auditable tables.

    All thresholds are configurable starting points, not calibrated probabilities.
    A None topic_mapping enables provisional cosine-based mapping suggestions.
    Pass a complete {topic_id: [theme_ids, ...]} mapping (including empty lists)
    after human review to compare approved topic meanings. No label cap is applied.
    Topic distributions use BERTopic's approximation, not cluster membership
    probabilities; long-text chunk scores are aggregated by maximum for both paths.

    Important parameters:
        columns: Text fields to pool for discovery, then summarize separately.
        themes: Theme ID -> description vocabulary for zero-shot and alignment.
        zero_threshold: Cutoff for each independently scored predefined theme.
        topic_threshold: Cutoff for each approximate discovered-topic weight.
        mapping_threshold: Cosine-similarity cutoff for suggested topic mappings.
            It has no effect on which topics the clustering discovers.
        min_topic_size: HDBSCAN minimum cluster size, not a requested topic count.
        cluster_selection_method: 'leaf' selects finer clusters; 'eom' favors
            broader stable groups in the HDBSCAN hierarchy.
        seed: Reproducibility setting, not a measure of stability across seeds.
        device: Transformers zero-shot device; -1 selects CPU.
        embedding_device: SentenceTransformer device; 'cpu' is the default.
        gold: Optional document ID -> human theme labels for evaluation only.

    Returns a dictionary of DataFrames (predictions, topics, topic_alignment,
    theme_distribution, coverage, agreement, gold_metrics), a metadata dictionary,
    and the fitted topic_model. Tables are returned in memory; the runner script
    handles saving them. Full scores remain available for inspecting thresholds.
    """
    # Step 1: check configuration before loading expensive model dependencies.
    for name, value in [('zero_threshold', zero_threshold),
                        ('topic_threshold', topic_threshold),
                        ('mapping_threshold', mapping_threshold)]:
        _threshold(value, name)
    if batch_size < 1 or min_topic_size < 2:
        raise ValueError('batch_size must be positive and min_topic_size at least two')
    import numpy as np
    import pandas as pd
    from transformers import set_seed
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer

    themes = THEME_DESCRIPTIONS if themes is None else themes
    if not themes or not all(isinstance(v, str) and v.strip() for v in themes.values()):
        raise ValueError('themes must map IDs to nonempty descriptions')
    if len(set(themes.values())) != len(themes):
        raise ValueError('Theme descriptions must be unique')
    if cluster_selection_method not in ('leaf', 'eom'):
        raise ValueError('cluster_selection_method must be leaf or eom')
    # Dictionary order provides a consistent theme order throughout the outputs.
    names = list(themes)
    documents = prepare_documents(df, columns)
    if len(documents) < max(10, min_topic_size + 1):
        raise ValueError('BERTopic comparison needs at least 10 nonblank review fields')
    # Seed model utilities and UMAP below. Exact results can still depend on the
    # package versions, hardware, input ordering, and numerical implementation.
    set_seed(seed)
    texts = documents.text.tolist()
    total_start = perf_counter()

    # Step 2: score the same documents against the caller's theme descriptions.
    # Classification, batching, and chunk-score aggregation live in zero-shot.py.
    zero_result = _zero_shot.classify_zero_shot(
        texts, themes, model=zero_model, batch_size=batch_size, device=device, seed=seed)
    zero_scores = zero_result['scores']
    zero_load_seconds = zero_result['load_seconds']
    zero_inference_seconds = zero_result['inference_seconds']

    # Step 3: generate semantic vectors without updating the encoder's weights.
    # Separate model-loading time from embedding, clustering, and scoring time.
    print('BERTopic: discovering topics with multilingual sentence embeddings...')
    start = perf_counter()
    encoder = SentenceTransformer(embedding_model, device=embedding_device)
    embedding_load_seconds = perf_counter() - start
    start = perf_counter()
    chunks, chunk_owners = [], []
    # Reserve two token positions for the encoder's usual boundary tokens.
    budget = max(1, encoder.max_seq_length - 2)
    for i, text in enumerate(texts):
        pieces = _chunks(text, encoder.tokenizer, budget)
        chunks.extend(pieces)
        # A flattened chunk batch needs an owner index to reconstruct each field.
        chunk_owners.extend([i] * len(pieces))
    chunk_embeddings = encoder.encode(chunks, batch_size=batch_size,
                                      normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.zeros((len(texts), chunk_embeddings.shape[1]))
    for i in range(len(texts)):
        # Every chunk has equal weight, including a shorter final chunk. Averaging
        # gives one vector per field but can blend distinct issues in long text.
        embeddings[i] = chunk_embeddings[np.array(chunk_owners) == i].mean(axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    # Normalize the averaged vectors again; the small floor avoids division by zero.

    # Step 4: fit an unsupervised topic model on all selected columns together.
    # Theme descriptions are deliberately absent from this clustering pipeline.
    model = BERTopic(
        embedding_model=encoder,
        # UMAP reduces vectors to 5 dimensions. A fixed seed and one worker help
        # reproduce runs; n_neighbors is capped for small document collections.
        umap_model=UMAP(n_neighbors=min(15, len(texts) - 1), n_components=5,
                        metric='cosine', random_state=seed, n_jobs=1),
        # HDBSCAN discovers the cluster count and can leave noise unassigned (-1).
        # min_samples controls density conservatism independently of cluster size.
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean',
                             min_samples=2, cluster_selection_method=cluster_selection_method,
                             prediction_data=True, core_dist_n_jobs=1),
        # Words and bigrams describe clusters after embedding. Stopword removal
        # here does not change transformer inputs; it is English-only.
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words='english'),
        # Class-based TF-IDF highlights terms distinctive to each topic and
        # reduces the influence of frequent words in the keyword representations.
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=False, verbose=False)
    primary_topics, _ = model.fit_transform(texts, embeddings)
    topic_info = model.get_topic_info()
    # -1 is the noise bucket, not a discovered topic to map to workplace themes.
    topic_ids = sorted(int(t) for t in topic_info.Topic if t != -1)
    topic_scores = [dict() for _ in texts]
    if topic_ids:
        # Step 5: estimate multiple topic weights per text using BERTopic's lexical
        # approximation. These are separate from HDBSCAN cluster memberships.
        # min_similarity filters window/topic matches internally; topic_threshold
        # later selects document labels from the resulting approximate weights.
        distributions, _ = model.approximate_distribution(chunks, min_similarity=0.1)
        if distributions.shape[1] != len(topic_ids):
            raise RuntimeError('Unexpected BERTopic distribution schema')
        for owner, distribution in zip(chunk_owners, distributions):
            for topic, score in zip(topic_ids, distribution):
                # A strong match in any chunk can retain a topic for the field.
                # Maxima across chunks need not sum to one and can favor long text.
                topic_scores[owner][topic] = max(topic_scores[owner].get(topic, 0), float(score))
    bertopic_seconds = perf_counter() - start

    # Step 6: align discovered topic keywords to the predefined theme vocabulary.
    # This post-processing never changes the clusters and is not a gold-label source.
    topic_texts = [' '.join(word for word, _ in model.get_topic(t)) for t in topic_ids]
    suggested_mapping, alignment_rows = {}, []
    if topic_ids:
        topic_vectors = encoder.encode(topic_texts, normalize_embeddings=True)
        theme_vectors = encoder.encode(list(themes.values()), normalize_embeddings=True)
        # With unit-length vectors, the dot product equals cosine similarity.
        # Shape: number of discovered topics x number of predefined themes.
        similarities = topic_vectors @ theme_vectors.T
        for i, topic in enumerate(topic_ids):
            scores = {name: float(similarities[i, j]) for j, name in enumerate(names)}
            suggested_mapping[topic] = select_labels(scores, mapping_threshold)
            alignment_rows.append({'topic_id': topic, 'keywords': topic_texts[i],
                                   'suggested_themes': suggested_mapping[topic],
                                   'cosine_scores': scores})
    if topic_mapping is None:
        mapping = suggested_mapping
        mapping_status = 'provisional_embedding_mapping'
    else:
        mapping = topic_mapping
        # Require an explicit decision for every topic, including [] for topics
        # outside the taxonomy. Missing entries must not silently discard topics.
        if set(mapping) != set(topic_ids):
            raise ValueError(f'Mapping must cover exactly the discovered topic IDs: {topic_ids}')
        if any(isinstance(v, str) for v in mapping.values()):
            raise ValueError('Each topic mapping must be a list of theme IDs')
        if any(set(labels) - set(names) for labels in mapping.values()):
            raise ValueError('Topic mapping contains unknown theme IDs')
        mapping_status = 'user_supplied_mapping'

    # Step 7: retain full scores plus thresholded labels for audit and plotting.
    # Primary clusters are single assignments; approximate labels can be multiple,
    # even for a field whose primary cluster is the -1 noise bucket.
    predictions = documents.copy()
    predictions['zero_shot_scores'] = zero_scores
    predictions['zero_shot_labels'] = [select_labels(s, zero_threshold) for s in zero_scores]
    predictions['bertopic_primary_topic'] = primary_topics
    predictions['bertopic_scores'] = topic_scores
    predictions['bertopic_labels'] = [select_labels(s, topic_threshold) for s in topic_scores]
    # Union all themes mapped from selected topics, without duplicate theme IDs.
    predictions['bertopic_theme_labels'] = [
        [name for name in names if any(name in mapping.get(t, []) for t in topics)]
        for topics in predictions.bertopic_labels]
    # Unmapped IDs are candidates for manual inspection, not proven novel themes.
    predictions['unmapped_topics'] = [[t for t in topics if not mapping.get(t)]
                                      for topics in predictions.bertopic_labels]
    # Compare only labels in the shared predefined vocabulary, not raw topic IDs.
    agreement = agreement_table(predictions.zero_shot_labels, predictions.bertopic_theme_labels)
    agreement.insert(0, 'document_id', documents.document_id.tolist())
    agreement.insert(1, 'column', documents.column.tolist())
    distribution_rows, summary_rows = [], []
    # Step 8: summarize each source column with its own nonblank-field denominator.
    # These are field counts, not unique-review counts across all four columns.
    for column in columns:
        subset = predictions[predictions.column == column]
        n = len(subset)
        for method, label_column in [('zero_shot', 'zero_shot_labels'),
                                      ('bertopic_mapped', 'bertopic_theme_labels'),
                                      ('bertopic_discovered', 'bertopic_labels')]:
            labels = subset[label_column]
            # Coverage is available for both raw topics and mapped themes. Mapping
            # can expand or remove labels, so those label counts are not equivalent.
            summary_rows.append({'column': column, 'method': method, 'n_documents': n,
                                 'unlabeled': sum(not x for x in labels),
                                 'multi_labeled': sum(len(x) > 1 for x in labels),
                                 'mean_labels': sum(map(len, labels)) / n if n else 0})
            if method == 'bertopic_discovered':
                # Raw topics do not share the predefined theme IDs used below.
                continue
            for theme in names:
                count = sum(theme in label_set for label_set in labels)
                # A field counts once per theme; overlapping percentages can exceed
                # 100% when added across themes. Empty columns report zero percent.
                distribution_rows.append({'column': column, 'method': method,
                                          'theme': theme, 'count': count,
                                          'pct': round(100 * count / n, 1) if n else 0})
    # Step 9: human annotations are the only reference for accuracy metrics.
    # Without gold labels the helper returns an empty metrics DataFrame.
    metrics = evaluate_predictions(predictions, gold or {}, names)
    # Record enough context to inspect runs; the hash covers prepared IDs/text in
    # order, not the full dataset or source code. Timings describe different tasks
    # in each path and should not be read as directly equivalent benchmarks.
    metadata = {
        'zero_model': zero_model, 'embedding_model': embedding_model,
        'themes': themes, 'zero_threshold': zero_threshold, 'topic_threshold': topic_threshold,
        'mapping_threshold': mapping_threshold, 'mapping_status': mapping_status,
        'topic_mapping': mapping, 'min_topic_size': min_topic_size, 'cluster_selection_method': cluster_selection_method,
        'min_samples': 2, 'seed': seed, 'zero_device': device,
        'embedding_device': embedding_device,
        'n_documents': len(documents), 'n_discovered_topics': len(topic_ids),
        'zero_load_seconds': zero_load_seconds, 'zero_inference_seconds': zero_inference_seconds,
        'embedding_load_seconds': embedding_load_seconds, 'bertopic_fit_and_score_seconds': bertopic_seconds,
        'total_seconds': perf_counter() - total_start,
        'input_sha256': sha256(documents[['document_id', 'text']].to_json().encode()).hexdigest(),
        'packages': {name: version(name) for name in ['transformers', 'sentence-transformers',
                                                     'bertopic', 'torch', 'scikit-learn']},
        'accuracy_status': 'human_annotations_supplied' if gold else 'not_measured',
        'decision': 'Review alignment, new topics and held-out F1 before selecting a winner.',
    }
    return {'predictions': predictions, 'topics': topic_info,
            'topic_alignment': pd.DataFrame(alignment_rows),
            'theme_distribution': pd.DataFrame(distribution_rows),
            'coverage': pd.DataFrame(summary_rows), 'agreement': agreement,
            'gold_metrics': metrics, 'metadata': metadata, 'topic_model': model}


def apply_topic_mapping(result, mapping, gold=None):
    """Recompute agreement and coverage after human mapping, without model reruns.

    Every discovered topic must have a list of approved predefined themes (or an
    empty list for a new theme). New themes remain visible as unmapped topic IDs.

    ``mapping`` uses topic IDs from this specific fitted result; IDs can change
    between fits. Return a new result dictionary with updated comparison tables.
    Existing embeddings, topic assignments, model, and score tables are reused.
    Pass gold again to evaluate the new mapping; old metrics are not preserved
    when gold is omitted. Suggested topic_alignment rows remain as an audit trail.
    """
    import pandas as pd
    names = list(result['metadata']['themes'])
    topic_ids = set(int(t) for t in result['topics'].Topic if t != -1)
    if set(mapping) != topic_ids:
        raise ValueError(f'Mapping must cover exactly these topic IDs: {sorted(topic_ids)}')
    if any(isinstance(labels, str) or set(labels) - set(names) for labels in mapping.values()):
        raise ValueError('Mapping values must be lists of known theme identifiers')
    # Copy containers that will change. The fitted model and unchanged tables are
    # shared with the original result; predictions and metadata are replaced.
    updated = dict(result)
    predictions = result['predictions'].copy()
    predictions['bertopic_theme_labels'] = [
        [name for name in names if any(name in mapping[t] for t in topics)]
        for topics in predictions.bertopic_labels]
    predictions['unmapped_topics'] = [[t for t in topics if not mapping[t]]
                                      for topics in predictions.bertopic_labels]
    # Recalculate every summary that depends on the approved topic meanings.
    agreement = agreement_table(predictions.zero_shot_labels, predictions.bertopic_theme_labels)
    agreement.insert(0, 'document_id', predictions.document_id.tolist())
    agreement.insert(1, 'column', predictions.column.tolist())
    distribution, coverage = [], []
    for column in result['coverage']['column'].unique():
        subset = predictions[predictions.column == column]
        n = len(subset)
        for method, label_column in [('zero_shot', 'zero_shot_labels'),
                                     ('bertopic_mapped', 'bertopic_theme_labels'),
                                      ('bertopic_discovered', 'bertopic_labels')]:
            labels = subset[label_column]
            coverage.append({'column': column, 'method': method, 'n_documents': n,
                             'unlabeled': sum(not x for x in labels),
                             'multi_labeled': sum(len(x) > 1 for x in labels),
                             'mean_labels': sum(map(len, labels)) / n if n else 0})
            if method == 'bertopic_discovered':
                continue
            for theme in names:
                count = sum(theme in values for values in labels)
                distribution.append({'column': column, 'method': method, 'theme': theme,
                                     'count': count, 'pct': round(100 * count / n, 1) if n else 0})
    updated.update(predictions=predictions, agreement=agreement,
                   theme_distribution=pd.DataFrame(distribution), coverage=pd.DataFrame(coverage),
                   gold_metrics=evaluate_predictions(predictions, gold or {}, names))
    updated['metadata'] = dict(result['metadata'], topic_mapping=mapping,
                               mapping_status='user_supplied_mapping',
                               accuracy_status='human_annotations_supplied' if gold else 'not_measured')
    return updated

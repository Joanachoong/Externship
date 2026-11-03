"""Compare multi-label NLI classification with BERTopic theme discovery.

Scores from the two methods have different meanings. Agreement is descriptive,
not accuracy. Discovered topics need human mapping before a validated comparison.
Transformer inputs preserve wording; the stemmed cleaner is used only by the
separate lexical analysis API. Model dependencies are loaded on demand.
"""

from hashlib import sha256
from importlib.metadata import version
from time import perf_counter

from .clean import prepare_semantic_text
from .theme_config import THEME_DESCRIPTIONS

ZERO_SHOT_MODEL = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TEXT_COLUMNS = ('review_cons', 'review_pros', 'summary', 'advice_to_management')



def _threshold(value, name):
    if not 0 < value <= 1:
        raise ValueError(f'{name} must be greater than zero and at most one')


def select_labels(scores, threshold):
    """Select every label meeting the threshold, without top-k or fallback."""
    _threshold(threshold, 'threshold')
    return [label for label, score in scores.items() if score >= threshold]


def prepare_documents(df, columns=TEXT_COLUMNS):
    """Return one row per non-null, nonblank review field with stable positional IDs."""
    import pandas as pd
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f'Missing review columns: {sorted(missing)}')
    rows = []
    for position in range(len(df)):
        row = df.iloc[position]
        for column in columns:
            value = row[column]
            if pd.isna(value):
                continue
            text = prepare_semantic_text(str(value))
            if text:
                rows.append({'document_id': f'{position}:{column}',
                             'row_position': position,
                             'review_id': row.get('review_id', position),
                             'column': column, 'text': text})
    return pd.DataFrame(rows, columns=['document_id', 'row_position', 'review_id',
                                      'column', 'text'])


def _chunks(text, tokenizer, token_budget):
    """Cover the whole text rather than silently truncating long reviews."""
    tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    pieces, start = [], 0
    while start < len(tokens):
        end = min(start + token_budget, len(tokens))
        piece = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        # Decoding a partial subword can slightly change re-tokenized length.
        while len(tokenizer.encode(piece, add_special_tokens=False, verbose=False)) > token_budget:
            end -= 1
            if end <= start:
                raise ValueError("A token cannot be represented within the chunk budget")
            piece = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        pieces.append(piece)
        start = end
    return pieces or [text]


def agreement_table(zero_sets, topic_sets):
    """Jaccard and exact set agreement; undefined Jaccard for two empty sets."""
    import pandas as pd
    if len(zero_sets) != len(topic_sets):
        raise ValueError('Prediction lists must refer to the same documents')
    rows = []
    for zero, topic in zip(zero_sets, topic_sets):
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
    """
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.preprocessing import MultiLabelBinarizer
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
    truth = encoder.fit_transform([gold[key] for key in reviewed.document_id])
    rows = []
    for method, column in [('zero_shot', 'zero_shot_labels'),
                           ('bertopic', 'bertopic_theme_labels')]:
        predicted = encoder.transform(reviewed[column])
        for average in ['micro', 'macro']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth, predicted, average=average, zero_division=0)
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
    """
    for name, value in [('zero_threshold', zero_threshold),
                        ('topic_threshold', topic_threshold),
                        ('mapping_threshold', mapping_threshold)]:
        _threshold(value, name)
    if batch_size < 1 or min_topic_size < 2:
        raise ValueError('batch_size must be positive and min_topic_size at least two')
    import numpy as np
    import pandas as pd
    from transformers import pipeline, set_seed
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
    names = list(themes)
    documents = prepare_documents(df, columns)
    if len(documents) < max(10, min_topic_size + 1):
        raise ValueError('BERTopic comparison needs at least 10 nonblank review fields')
    set_seed(seed)
    texts = documents.text.tolist()
    total_start = perf_counter()

    print(f'Zero-shot: scoring {len(texts)} fields against {len(names)} themes...')
    start = perf_counter()
    classifier = pipeline('zero-shot-classification', model=zero_model, device=device)
    zero_load_seconds = perf_counter() - start
    start = perf_counter()
    zero_scores = []
    zero_chunks = []
    owners = []
    for i, text in enumerate(texts):
        chunks = _chunks(text, classifier.tokenizer, 256)
        zero_chunks.extend(chunks)
        owners.extend([i] * len(chunks))
    accumulated = [{name: 0.0 for name in names} for _ in texts]
    description_to_name = {description: name for name, description in themes.items()}
    for offset in range(0, len(zero_chunks), batch_size):
        outputs = classifier(zero_chunks[offset:offset + batch_size],
                             candidate_labels=list(themes.values()), multi_label=True,
                             hypothesis_template='This text discusses {}.',
                             batch_size=batch_size)
        for owner, output in zip(owners[offset:offset + batch_size], outputs):
            for label, score in zip(output['labels'], output['scores']):
                name = description_to_name[label]
                accumulated[owner][name] = max(accumulated[owner][name], float(score))
    zero_scores = accumulated
    zero_inference_seconds = perf_counter() - start
    del classifier

    print('BERTopic: discovering topics with multilingual sentence embeddings...')
    start = perf_counter()
    encoder = SentenceTransformer(embedding_model, device=embedding_device)
    embedding_load_seconds = perf_counter() - start
    start = perf_counter()
    chunks, chunk_owners = [], []
    budget = max(1, encoder.max_seq_length - 2)
    for i, text in enumerate(texts):
        pieces = _chunks(text, encoder.tokenizer, budget)
        chunks.extend(pieces)
        chunk_owners.extend([i] * len(pieces))
    chunk_embeddings = encoder.encode(chunks, batch_size=batch_size,
                                      normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.zeros((len(texts), chunk_embeddings.shape[1]))
    for i in range(len(texts)):
        embeddings[i] = chunk_embeddings[np.array(chunk_owners) == i].mean(axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-12)
    model = BERTopic(
        embedding_model=encoder,
        umap_model=UMAP(n_neighbors=min(15, len(texts) - 1), n_components=5,
                        metric='cosine', random_state=seed, n_jobs=1),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean',
                             min_samples=2, cluster_selection_method=cluster_selection_method,
                             prediction_data=True, core_dist_n_jobs=1),
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words='english'),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        calculate_probabilities=False, verbose=False)
    primary_topics, _ = model.fit_transform(texts, embeddings)
    topic_info = model.get_topic_info()
    topic_ids = sorted(int(t) for t in topic_info.Topic if t != -1)
    topic_scores = [dict() for _ in texts]
    if topic_ids:
        distributions, _ = model.approximate_distribution(chunks, min_similarity=0.1)
        if distributions.shape[1] != len(topic_ids):
            raise RuntimeError('Unexpected BERTopic distribution schema')
        for owner, distribution in zip(chunk_owners, distributions):
            for topic, score in zip(topic_ids, distribution):
                topic_scores[owner][topic] = max(topic_scores[owner].get(topic, 0), float(score))
    bertopic_seconds = perf_counter() - start

    # Alignment is an explicit approximation, not a new gold-label source.
    topic_texts = [' '.join(word for word, _ in model.get_topic(t)) for t in topic_ids]
    suggested_mapping, alignment_rows = {}, []
    if topic_ids:
        topic_vectors = encoder.encode(topic_texts, normalize_embeddings=True)
        theme_vectors = encoder.encode(list(themes.values()), normalize_embeddings=True)
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
        if set(mapping) != set(topic_ids):
            raise ValueError(f'Mapping must cover exactly the discovered topic IDs: {topic_ids}')
        if any(isinstance(v, str) for v in mapping.values()):
            raise ValueError('Each topic mapping must be a list of theme IDs')
        if any(set(labels) - set(names) for labels in mapping.values()):
            raise ValueError('Topic mapping contains unknown theme IDs')
        mapping_status = 'user_supplied_mapping'

    predictions = documents.copy()
    predictions['zero_shot_scores'] = zero_scores
    predictions['zero_shot_labels'] = [select_labels(s, zero_threshold) for s in zero_scores]
    predictions['bertopic_primary_topic'] = primary_topics
    predictions['bertopic_scores'] = topic_scores
    predictions['bertopic_labels'] = [select_labels(s, topic_threshold) for s in topic_scores]
    predictions['bertopic_theme_labels'] = [
        [name for name in names if any(name in mapping.get(t, []) for t in topics)]
        for topics in predictions.bertopic_labels]
    predictions['unmapped_topics'] = [[t for t in topics if not mapping.get(t)]
                                      for topics in predictions.bertopic_labels]
    agreement = agreement_table(predictions.zero_shot_labels, predictions.bertopic_theme_labels)
    agreement.insert(0, 'document_id', documents.document_id.tolist())
    agreement.insert(1, 'column', documents.column.tolist())
    distribution_rows, summary_rows = [], []
    for column in columns:
        subset = predictions[predictions.column == column]
        n = len(subset)
        for method, label_column in [('zero_shot', 'zero_shot_labels'),
                                      ('bertopic_mapped', 'bertopic_theme_labels'),
                                      ('bertopic_discovered', 'bertopic_labels')]:
            labels = subset[label_column]
            summary_rows.append({'column': column, 'method': method, 'n_documents': n,
                                 'unlabeled': sum(not x for x in labels),
                                 'multi_labeled': sum(len(x) > 1 for x in labels),
                                 'mean_labels': sum(map(len, labels)) / n if n else 0})
            if method == 'bertopic_discovered':
                continue
            for theme in names:
                count = sum(theme in label_set for label_set in labels)
                distribution_rows.append({'column': column, 'method': method,
                                          'theme': theme, 'count': count,
                                          'pct': round(100 * count / n, 1) if n else 0})
    metrics = evaluate_predictions(predictions, gold or {}, names)
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
    """
    import pandas as pd
    names = list(result['metadata']['themes'])
    topic_ids = set(int(t) for t in result['topics'].Topic if t != -1)
    if set(mapping) != topic_ids:
        raise ValueError(f'Mapping must cover exactly these topic IDs: {sorted(topic_ids)}')
    if any(isinstance(labels, str) or set(labels) - set(names) for labels in mapping.values()):
        raise ValueError('Mapping values must be lists of known theme identifiers')
    updated = dict(result)
    predictions = result['predictions'].copy()
    predictions['bertopic_theme_labels'] = [
        [name for name in names if any(name in mapping[t] for t in topics)]
        for topics in predictions.bertopic_labels]
    predictions['unmapped_topics'] = [[t for t in topics if not mapping[t]]
                                      for topics in predictions.bertopic_labels]
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

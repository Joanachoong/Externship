"""Score predefined workplace themes using multilingual zero-shot NLI.

Import ``classify_zero_shot`` from ``text_engine`` to use this hyphenated module.
The model is loaded only when classification is requested. Text should preserve
its natural wording, punctuation, and negation rather than being stemmed.
"""

from time import perf_counter

from ._semantic import _chunks
from .theme_config import THEME_DESCRIPTIONS

# This pretrained natural language inference (NLI) model evaluates whether a
# review supports a statement about a candidate theme. No local fine-tuning occurs.
ZERO_SHOT_MODEL = 'MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli'


def classify_zero_shot(texts, themes=None, *, model=ZERO_SHOT_MODEL,
                       batch_size=8, device=-1, seed=42):
    """Return per-text theme scores plus model-loading and inference timings.

    ``texts`` is an iterable of natural-language strings. ``themes`` maps theme
    IDs to unique descriptions; the configured workplace themes are the default.
    Each candidate is scored independently, retaining the maximum score across
    a text's tokenizer chunks. Scores are not calibrated probabilities. This
    function does not apply a label threshold or discover new themes.

    Parameters:
        texts: Input strings, ordered as the desired output; do not pass one
            bare string. Unlike prepare_documents(), this API does not drop blanks.
        themes: Theme IDs mapped to descriptions inserted into the hypothesis
            template, for example {'management': 'management and leadership'}.
        model: Hugging Face checkpoint name or local model path.
        batch_size: Maximum number of text chunks sent per classifier call.
        device: Transformers pipeline device; -1 uses CPU.
        seed: Random seed passed to Transformers before model initialization.

    The result's ``scores`` list has one {theme_id: score} dictionary per input.
    ``load_seconds`` measures pipeline creation and ``inference_seconds`` includes
    token chunking and scoring. Empty input returns empty scores and zero timings
    without importing Transformers or loading model weights.
    """
    if isinstance(texts, str):
        raise ValueError('texts must be an iterable of strings, not a single string')
    # Materialize generators once so indexing and output ownership stay stable.
    texts = list(texts)
    if not all(isinstance(text, str) for text in texts):
        raise ValueError('Every input text must be a string')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    themes = THEME_DESCRIPTIONS if themes is None else themes
    # Descriptions must be unique because pipeline results identify candidates by
    # their description strings, which we map back to application theme IDs.
    if not themes or not all(isinstance(v, str) and v.strip() for v in themes.values()):
        raise ValueError('themes must map IDs to nonempty descriptions')
    if len(set(themes.values())) != len(themes):
        raise ValueError('Theme descriptions must be unique')
    if not texts:
        return {'scores': [], 'load_seconds': 0.0, 'inference_seconds': 0.0}

    # Keep this import local: merely importing the package must not initialize
    # model dependencies. A first real call may download uncached model weights.
    from transformers import pipeline, set_seed

    set_seed(seed)
    print(f'Zero-shot: scoring {len(texts)} fields against {len(themes)} themes...')
    start = perf_counter()
    classifier = pipeline('zero-shot-classification', model=model, device=device)
    load_seconds = perf_counter() - start
    start = perf_counter()
    chunks, owners = [], []
    for i, text in enumerate(texts):
        # Use a conservative content budget to leave space for the paired theme
        # hypothesis and special tokens. This assumes normal short descriptions.
        pieces = _chunks(text, classifier.tokenizer, 256)
        chunks.extend(pieces)
        owners.extend([i] * len(pieces))
    # owners[k] tells us which original input produced flattened chunk k.
    scores = [{name: 0.0 for name in themes} for _ in texts]
    description_to_name = {description: name for name, description in themes.items()}
    for offset in range(0, len(chunks), batch_size):
        # With multi_label=True each theme is evaluated independently: two themes
        # can both score highly. This is theme relevance, not sentiment polarity.
        outputs = classifier(chunks[offset:offset + batch_size],
                             candidate_labels=list(themes.values()), multi_label=True,
                             hypothesis_template='This text discusses {}.',
                             batch_size=batch_size)
        for owner, output in zip(owners[offset:offset + batch_size], outputs):
            # The pipeline sorts candidates by score, so match labels explicitly
            # rather than assuming its output order equals dictionary order.
            for label, score in zip(output['labels'], output['scores']):
                name = description_to_name[label]
                # Keep a theme's strongest supporting chunk. These maxima are
                # not a distribution summing to one; long fields get more chances
                # to exceed a downstream cutoff than short fields.
                scores[owner][name] = max(scores[owner][name], float(score))
    return {'scores': scores, 'load_seconds': load_seconds,
            'inference_seconds': perf_counter() - start}

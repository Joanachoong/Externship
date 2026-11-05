"""
NLP pipeline: preprocessing, linguistic analysis, text representation.

Operations:
1. Lowercase and remove punctuation
2. Split text into words
3. Remove stopwords
4. Lemmatization - for (WordNet)
5. Stemming (Snowball)
6. POS tagging (on raw text)
7. Syntax (dependency) parsing (on raw text)
8. Text representation: BoW and TF-IDF

The helpers are separate operations: clean() does not automatically run POS
tagging, syntax parsing, or vectorization. prepare_semantic_text() provides a
lighter path for transformers that preserves the original language structure.

Stemming removes meaning from words, e.g.
"Classifying product reviews as positive, or negative..."
-> "classifi product review posit negat base densiti word like"
"""

import re
import string
from functools import lru_cache
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import nltk
# str.translate deletes these punctuation characters; it does not insert spaces.
# For example, 'fast-paced' becomes 'fastpaced' in the lexical pipeline.
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def download_nltk_resources(download_dir=None):
    """Explicitly install resources for cleaning and POS tagging.

    No resources are downloaded merely by importing this module. Pass a
    writable download directory when the default NLTK location is unavailable.
    """
    import nltk
    # Register a custom location before downloading so later NLTK lookups find it.
    if download_dir and download_dir not in nltk.data.path:
        nltk.data.path.insert(0, str(download_dir))
    for package in ('stopwords', 'wordnet', 'punkt_tab',
                    'averaged_perceptron_tagger_eng'):
        # Resources serve cleaning, sentence/word tokenization, and English POS
        # tagging respectively. Downloading is an explicit setup step, not import
        # behavior, so offline analysis fails with a clear missing-data error.
        if not nltk.download(package, download_dir=download_dir, quiet=True):
            raise RuntimeError(f"Could not download NLTK resource: {package}")


@lru_cache(maxsize=1)
def _cleaning_resources():
    """Load and cache the English lexical tools after verifying NLTK data exists."""

    try:
        stops = set(stopwords.words('english'))
        lemma = WordNetLemmatizer()
        # WordNet loads lazily; one probe forces a missing corpus to fail here
        # rather than halfway through analyzing a DataFrame.
        lemma.lemmatize('words')
    except LookupError as exc:
        raise LookupError(
            "Missing NLTK data. Call download_nltk_resources() once before cleaning."
        ) from exc
    return stops, SnowballStemmer('english'), lemma


def prepare_semantic_text(text):
    """Normalize whitespace while preserving case, punctuation and negation.

    Use this input for transformer models, not the stemmed output of clean().
    Tabs/newlines/repeated spaces become one space; outer whitespace is removed.
    The helper does not translate, remove stopwords, or validate text language.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", text).strip()


# ---------- Preprocessing ----------
def clean(text):
    """Return stopword-filtered, lemmatized, stemmed English tokens.

    Intended for word counts, lexical keyword matching, and BoW/TF-IDF features.
    Stopword removal may discard negation and stemming loses natural word forms,
    so this output should not be used as transformer/sentiment model input.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    STOP_WORDS, stemmer, lemmatizer = _cleaning_resources()
    # Normalize surface form, then tokenize on whitespace rather than using a
    # punctuation-aware tokenizer. No spelling correction or language detection.
    text = text.lower().translate(PUNCT_TABLE)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    # WordNet receives no POS argument, so noun lemmatization is used by default.
    # Snowball then reduces the result further, often to a non-dictionary stem.
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)


# ---------- POS tagging (run on RAW text) ----------
def pos_tag_text(text):
    """Return NLTK (token, English POS-tag) pairs from the original text.

    Word tokenization retains useful punctuation/context that clean() removes.
    Requires the tokenizer and English tagger resources installed by the setup
    helper; these tags are not theme labels or sentiment predictions.
    """

    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)          # [('There', 'EX'), ('is', 'VBZ'), ...]


# ---------- Syntax parsing (run on RAW text) ----------
# Use spaCy's installed English dependency pipeline for grammatical relations.

@lru_cache(maxsize=1)
def _syntax_model():
    """Load spaCy's English pipeline once, only when syntax analysis is requested.

    en_core_web_sm must already be installed; this does not download the model.
    """
    import spacy
    return spacy.load("en_core_web_sm")


def parse_syntax(text):
    """Return raw-text (token, POS, dependency relation, head token) tuples.

    The dependency relation links a token to its grammatical head. This helper
    exposes linguistic structure for inspection; the theme engine does not call it.
    """
    nlp = _syntax_model()
    doc = nlp(text)
    return [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]
    # e.g. ('thinking', 'VERB', 'csubj', 'makes')


# ---------- Text representation ----------
def text_representation(corpus):
    """
    BoW: counts how many times each word appears in a document.
    TF-IDF: weighs each word by how frequent it is in a document versus
            how rare it is across the corpus, so common words matter less.

    ``corpus`` is an iterable of strings. Return two document-by-word DataFrames
    and their fitted vectorizers. Row order follows the input corpus. These are
    lexical features, not contextual sentence embeddings. An empty corpus or an
    all-filtered vocabulary raises the vectorizer's ValueError.
    """

    cleaned = [clean(doc) for doc in corpus]

    # Fit both vocabularies on this corpus. Defaults use unigrams and may ignore
    # single-character tokens; transformer tokenization is a different operation.
    bow = CountVectorizer()                      # add ngram_range=(1, 2) for bigrams
    X_bow = bow.fit_transform(cleaned)

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(cleaned)

    # Dense conversion is convenient for small notebooks but may use substantial
    # memory on large corpora. TF-IDF is rounded only in the returned display table;
    # the fitted vectorizer retains its full-precision weighting parameters.
    bow_df = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out()).round(3)
    return bow_df, tfidf_df, bow, tfidf


if __name__ == "__main__":
    # Manual demonstration only; importing clean.py does not run these examples.
    # Install the required NLTK resources and spaCy model before executing it.
    corpus = [
        "Classifying product reviews as positive, or negative based on the density of words like",
        "There is nothing either good or bad but thinking makes it so.",
    ]

    print(pos_tag_text(corpus[1]))
    print(parse_syntax(corpus[1]))

    bow_df, tfidf_df, bow, tfidf = text_representation(corpus)
    print(bow_df)
    print(tfidf_df)

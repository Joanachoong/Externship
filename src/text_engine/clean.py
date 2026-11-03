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

LIMITATION TO FLAG
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
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def download_nltk_resources(download_dir=None):
    """Explicitly install resources for cleaning and POS tagging.

    No resources are downloaded merely by importing this module. Pass a
    writable download directory when the default NLTK location is unavailable.
    """
    import nltk
    if download_dir and download_dir not in nltk.data.path:
        nltk.data.path.insert(0, str(download_dir))
    for package in ('stopwords', 'wordnet', 'punkt_tab',
                    'averaged_perceptron_tagger_eng'):
        if not nltk.download(package, download_dir=download_dir, quiet=True):
            raise RuntimeError(f"Could not download NLTK resource: {package}")


@lru_cache(maxsize=1)
def _cleaning_resources():

    try:
        stops = set(stopwords.words('english'))
        lemma = WordNetLemmatizer()
        lemma.lemmatize('words')
    except LookupError as exc:
        raise LookupError(
            "Missing NLTK data. Call download_nltk_resources() once before cleaning."
        ) from exc
    return stops, SnowballStemmer('english'), lemma


def prepare_semantic_text(text):
    """Normalize whitespace while preserving case, punctuation and negation.

    Use this input for transformer models, not the stemmed output of clean().
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.sub(r"\s+", " ", text).strip()


# ---------- Preprocessing ----------
def clean(text):
    """Return stopword-filtered, lemmatized, stemmed English tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    STOP_WORDS, stemmer, lemmatizer = _cleaning_resources()
    text = text.lower().translate(PUNCT_TABLE)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)


# ---------- POS tagging (run on RAW text) ----------
def pos_tag_text(text):

    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)          # [('There', 'EX'), ('is', 'VBZ'), ...]


# ---------- Syntax parsing (run on RAW text) ----------
# NLTK has no good pretrained parser, so use spaCy:

@lru_cache(maxsize=1)
def _syntax_model():
    import spacy
    return spacy.load("en_core_web_sm")


def parse_syntax(text):
    """Return raw-text token, POS, dependency, and head tuples."""
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
    """

    cleaned = [clean(doc) for doc in corpus]

    bow = CountVectorizer()                      # add ngram_range=(1, 2) for bigrams
    X_bow = bow.fit_transform(cleaned)

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(cleaned)

    bow_df = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out()).round(3)
    return bow_df, tfidf_df, bow, tfidf


if __name__ == "__main__":
    corpus = [
        "Classifying product reviews as positive, or negative based on the density of words like",
        "There is nothing either good or bad but thinking makes it so.",
    ]

    print(pos_tag_text(corpus[1]))
    print(parse_syntax(corpus[1]))

    bow_df, tfidf_df, bow, tfidf = text_representation(corpus)
    print(bow_df)
    print(tfidf_df)

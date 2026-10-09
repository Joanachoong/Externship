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

import spacy
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

for pkg in ['stopwords', 'wordnet', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


# ---------- Preprocessing ----------
def clean(text):
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

def parse_syntax(text):
   
    nlp = spacy.load("en_core_web_sm")
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
"""
This python folder will clean all the text before we move on to the next step
"""

import nltk
nltk.download('stopwords')

import string
from nltk.corpus import stopwords

def clean(text):
    # Convert to lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))

    # Split into words
    words = text.split()

    # Remove stopwords
    filtered = [word for word in words if word not in stopwords.words('english')]

    # Join back into a single string
    return ' '.join(filtered)
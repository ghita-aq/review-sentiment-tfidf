"""
Text preprocessing utilities for the Review Sentiment Classifier.

Provides functions for cleaning, tokenizing, removing stopwords, and lemmatizing
customer review text before feeding it into the TF-IDF vectorizer.
"""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data on first import
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Pre-compute reusable objects
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def remove_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, ftp) from text."""
    return re.sub(r"https?://\S+|ftp://\S+", "", text)


def remove_special_chars(text: str) -> str:
    """Remove special characters and digits, keeping only letters and whitespace."""
    return re.sub(r"[^a-zA-Z\s]", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from tokenized text."""
    tokens = text.split()
    filtered = [token for token in tokens if token not in _STOPWORDS]
    return " ".join(filtered)


def lemmatize(text: str) -> str:
    """Lemmatize each token in the text using WordNet."""
    tokens = text.split()
    lemmatized = [_LEMMATIZER.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)


def clean_text(text: str) -> str:
    """
    Apply basic text cleaning: remove HTML, URLs, special characters,
    normalize whitespace, and lowercase.
    """
    if not text or not isinstance(text, str):
        return ""
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_chars(text)
    text = remove_extra_whitespace(text)
    text = lowercase(text)
    return text


def preprocess_review(text: str) -> str:
    """
    Full preprocessing pipeline for a single review.

    Steps:
      1. Clean text (HTML, URLs, special chars, whitespace, lowercase)
      2. Remove stopwords
      3. Lemmatize

    Returns a clean, space-separated string of processed tokens.
    """
    if not text or not isinstance(text, str):
        return ""

    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = remove_extra_whitespace(text)

    return text


def preprocess_batch(reviews: list) -> list:
    """Apply the full preprocessing pipeline to a list of reviews."""
    return [preprocess_review(review) for review in reviews]


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_reviews = [
        "This product is AMAZING!!! Check it out at http://example.com",
        "<p>Terrible service</p>",
        "The item was okay, nothing special. It works as expected I guess.",
        "Absolutely wonderful experience! Highly recommend to everyone!!",
        "Worst purchase ever. Broke after one day. Do NOT buy this.",
    ]

    print("Preprocessing Examples\n", "=" * 60)
    for review in sample_reviews:
        cleaned = preprocess_review(review)
        print(f"\nOriginal : {review}")
        print(f"Processed: {cleaned}")

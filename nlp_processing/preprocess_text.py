"""
nlp_processing/preprocess_text.py
-----------------------------------
NLP preprocessing pipeline using NLTK.

Responsibilities:
  - Tokenise text into words.
  - Remove stopwords and punctuation.
  - Perform lemmatisation.
  - Extract keywords from a sentence using TF-IDF scores.
  - Score sentences by importance (keyword density).

All NLTK data is downloaded automatically on first run.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# ── Auto-download required NLTK corpora ───────────────────────────
for _pkg in ("punkt", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────
# Token-level helpers
# ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercased word tokens, no punctuation."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in string.punctuation]


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter NLTK English stopwords."""
    return [t for t in tokens if t not in _STOP_WORDS]


def lemmatize(tokens: list[str]) -> list[str]:
    """Reduce each token to its base lemma."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_sentence(sentence: str) -> list[str]:
    """
    Full pipeline for a single sentence.
    Returns cleaned, lemmatised tokens.
    """
    tokens = tokenize(sentence)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    # Keep only alphabetic tokens of length >= 3
    return [t for t in tokens if t.isalpha() and len(t) >= 3]


# ─────────────────────────────────────────────────────────────────
# Sentence-level helpers
# ─────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Use NLTK sentence tokeniser."""
    return sent_tokenize(text)


def extract_keywords(sentence: str, top_n: int = 5) -> list[str]:
    """
    Return the top-N most informative words from a sentence.
    Uses simple frequency after stopword removal (no TF-IDF needed
    for single-sentence keyword extraction).
    """
    tokens = preprocess_sentence(sentence)
    if not tokens:
        return []
    # Frequency-based ranking (descending)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq, key=freq.get, reverse=True)
    return ranked[:top_n]


def score_sentences(sentences: list[str], keywords: set[str]) -> list[tuple[float, str]]:
    """
    Score each sentence by the fraction of its tokens that are
    in the global keyword set.  Returns list of (score, sentence)
    sorted descending.
    """
    scored: list[tuple[float, str]] = []
    for sent in sentences:
        tokens = preprocess_sentence(sent)
        if not tokens:
            continue
        hit = sum(1 for t in tokens if t in keywords)
        score = hit / len(tokens)
        scored.append((score, sent))
    return sorted(scored, reverse=True)


# ─────────────────────────────────────────────────────────────────
# Batch pipeline
# ─────────────────────────────────────────────────────────────────

def preprocess_corpus(sentences: list[str]) -> dict:
    """
    Process a list of sentences and return:
      - preprocessed : list of token lists (one per sentence)
      - all_keywords : flat set of all extracted keywords
      - scored       : sentences ranked by keyword density
    """
    preprocessed: list[list[str]] = []
    all_keywords: set[str] = set()

    for sent in sentences:
        tokens = preprocess_sentence(sent)
        preprocessed.append(tokens)
        kw = extract_keywords(sent)
        all_keywords.update(kw)

    scored = score_sentences(sentences, all_keywords)

    print(f"[NLP] Preprocessed {len(sentences)} sentences, "
          f"{len(all_keywords)} unique keywords discovered.")

    return {
        "preprocessed": preprocessed,
        "all_keywords": all_keywords,
        "scored_sentences": scored,
    }


# ─────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "TCP/IP is the fundamental communication protocol of the Internet.",
        "Binary search finds an element in a sorted array in O(log n) time.",
        "This chapter introduces the topic discussed in later sections.",
    ]
    result = preprocess_corpus(sample)
    print("Keywords:", result["all_keywords"])
    print("Scored sentences:")
    for score, s in result["scored_sentences"]:
        print(f"  [{score:.2f}] {s}")

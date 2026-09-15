"""
Robust Text Preprocessing Pipeline for SentimentIQ.
Ensures consistent text normalization across Training, Evaluation, Testing, and Production Inference.
"""

import re
import html
from typing import List, Union

# Common English Contractions Map (focusing on preserving negation integrity)
CONTRACTION_MAP = {
    r"won\'t": "will not",
    r"can\'t": "can not",
    r"cannot": "can not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'t": " not",
    r"\'ve": " have",
    r"\'m": " am",
}

# Negation and critical sentiment-modifying keywords to protect
SENTIMENT_KEYWORDS = {
    "not", "no", "never", "nor", "neither", "none",
    "hardly", "scarcely", "barely", "little", "few"
}


def clean_text(text: Union[str, float, None]) -> str:
    """
    Clean and normalize an individual text string for sentiment analysis.
    
    Processing Steps:
    1. Type validation and null check
    2. HTML unescaping and tag stripping
    3. URL and link removal
    4. Contraction expansion (especially preserving negations like 'not', 'never')
    5. Lowercase normalization
    6. Non-alphanumeric character removal (preserving words and whitespace)
    7. Extra whitespace collapsing and trimming

    Args:
        text: Raw input review string.

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""
    
    # Convert non-string types safely
    if not isinstance(text, str):
        text = str(text)
    
    # Unescape HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = html.unescape(text)
    
    # Remove HTML tags (<p>, <br />, <div>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Remove URLs and links (http, https, www, ftp)
    text = re.sub(r"https?://\S+|www\.\S+|ftp://\S+", " ", text)
    
    # Remove email addresses
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    
    # Lowercase text early for uniform regex matching
    text = text.lower()
    
    # Expand contractions (critical for sentiment negation: "didn't" -> "did not")
    for pattern, replacement in CONTRACTION_MAP.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove special characters, symbols, emojis, and digits, retaining letters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)
    
    # Collapse multiple whitespaces and tabs into a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def preprocess_corpus(texts: List[Union[str, None]]) -> List[str]:
    """
    Preprocess a list or Series of text strings in batch.

    Args:
        texts: Collection of raw review strings.

    Returns:
        List of cleaned strings.
    """
    return [clean_text(t) for t in texts]


def extract_sentiment_tokens(text: str) -> List[str]:
    """
    Tokenize preprocessed text into individual words.
    
    Args:
        text: Input string.
        
    Returns:
        List of word tokens.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()

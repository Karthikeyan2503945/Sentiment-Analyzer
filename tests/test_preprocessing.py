"""
Unit Tests for Text Preprocessing and Negation Preservation Pipeline.
"""

import pytest
from src.preprocessing import clean_text, preprocess_corpus, extract_sentiment_tokens


def test_lowercase_conversion():
    raw = "AMAZING Experience with Great Customer Support!"
    expected = "amazing experience with great customer support"
    assert clean_text(raw) == expected


def test_html_tag_removal():
    raw = "<p>The item was <b>awesome</b> and <br />arrived quickly!</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "<br />" not in cleaned
    assert "the item was awesome and arrived quickly" == cleaned


def test_url_and_email_removal():
    raw = "Check out my review at https://reviews.example.com/item and email me at test@example.com!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "reviews.example.com" not in cleaned
    assert "test@example.com" not in cleaned
    assert "check out my review at and email me at" == cleaned


def test_negation_preservation():
    # Crucial test: Negation words must not be stripped or lost
    raw_1 = "I don't like this phone, it wasn't good at all."
    cleaned_1 = clean_text(raw_1)
    assert "not" in cleaned_1
    assert "i do not like this phone it was not good at all" == cleaned_1

    raw_2 = "Never buy this, it can't perform properly."
    cleaned_2 = clean_text(raw_2)
    assert "never" in cleaned_2
    assert "not" in cleaned_2
    assert "never buy this it can not perform properly" == cleaned_2


def test_empty_and_null_inputs():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_punctuation_and_special_characters():
    raw = "Superb battery!!! ($99.99)... #BestBuy @Home ~100% 👍"
    cleaned = clean_text(raw)
    assert "!" not in cleaned
    assert "$" not in cleaned
    assert "#" not in cleaned
    assert "@" not in cleaned
    assert "%" not in cleaned
    assert "superb battery bestbuy home" == cleaned


def test_very_long_input():
    long_review = "Great product! " * 500
    cleaned = clean_text(long_review)
    assert len(cleaned) > 0
    assert cleaned.startswith("great product")


def test_batch_preprocessing():
    texts = [
        "<b>Good</b> product!",
        "Terrible experience: http://bad.com",
        None,
        "Neutral delivery."
    ]
    results = preprocess_corpus(texts)
    assert len(results) == 4
    assert results[0] == "good product"
    assert results[1] == "terrible experience"
    assert results[2] == ""
    assert results[3] == "neutral delivery"


def test_extract_sentiment_tokens():
    tokens = extract_sentiment_tokens("Fast delivery, great service!")
    assert tokens == ["fast", "delivery", "great", "service"]

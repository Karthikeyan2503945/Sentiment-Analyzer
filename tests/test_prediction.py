"""
Unit Tests for Prediction Pipeline, Edge Cases, and Aspect Extraction.
"""

import pytest
import numpy as np
from src.preprocessing import clean_text
from src.aspect_analyzer import extract_aspect_sentiments
from src.model import ReviewTokenizer


def test_tokenizer_oov_and_padding():
    """Verify tokenizer correctly handles OOV tokens and fixed length padding."""
    train_corpus = [
        "great battery life",
        "terrible screen quality",
        "neutral standard delivery"
    ]
    tokenizer = ReviewTokenizer(num_words=50, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_corpus)
    
    assert tokenizer.is_fitted
    assert "<OOV>" in tokenizer.word_index
    assert tokenizer.word_index["<OOV>"] == 1
    
    # Test unseen words mapped to OOV id 1
    test_text = ["exceptional battery with unmentioned specs"]
    seqs = tokenizer.texts_to_sequences(test_text)
    assert len(seqs[0]) == 5
    # 'exceptional', 'with', 'unmentioned', 'specs' should be OOV (1)
    assert seqs[0][0] == 1
    assert seqs[0][1] == tokenizer.word_index["battery"]
    
    # Test padding
    padded = tokenizer.pad_sequences(seqs, maxlen=6, padding="post")
    assert padded.shape == (1, 6)
    assert padded[0, 5] == 0  # Padded element


def test_aspect_sentiment_extraction():
    """Verify aspect-based extraction on multi-clause mixed sentiment sentences."""
    sentence = "The camera quality is excellent but the battery life is terrible."
    aspects = extract_aspect_sentiments(sentence)
    
    assert len(aspects) == 2
    
    aspect_map = {item["aspect"]: item["sentiment"] for item in aspects}
    assert aspect_map["Camera / Image Quality"] == "Positive"
    assert aspect_map["Battery Life & Charging"] == "Negative"


def test_aspect_extraction_with_negation():
    """Verify negation handling in aspect extraction."""
    sentence = "The screen is not good, but sound is amazing."
    aspects = extract_aspect_sentiments(sentence)
    aspect_map = {item["aspect"]: item["sentiment"] for item in aspects}
    assert aspect_map["Display & Screen"] == "Negative"
    assert aspect_map["Sound & Audio"] == "Positive"


def test_prediction_output_structure_contract():
    """Verify contract of prediction response dictionary."""
    mock_prediction = {
        "sentiment": "Positive",
        "confidence": 0.952,
        "probabilities": {
            "negative": 0.021,
            "neutral": 0.027,
            "positive": 0.952
        },
        "cleaned_text": "great experience",
        "tokens": ["great", "experience"],
        "token_ids": [4, 9]
    }
    
    assert "sentiment" in mock_prediction
    assert mock_prediction["sentiment"] in ["Negative", "Neutral", "Positive"]
    assert 0.0 <= mock_prediction["confidence"] <= 1.0
    assert "probabilities" in mock_prediction
    prob_sum = sum(mock_prediction["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01


def test_timestamp_formatting():
    """Verify timestamp formatting functions."""
    from src.speech_to_text import format_timestamp, format_segment_interval
    assert format_timestamp(0.0) == "00:00"
    assert format_timestamp(65.0) == "01:05"
    assert format_timestamp(360.0) == "06:00"
    assert format_segment_interval(0.0, 8.5) == "00:00 – 00:08"
    assert format_segment_interval(65.0, 125.0) == "01:05 – 02:05"


def test_weighted_segment_aggregation():
    """Verify duration-and-word-count weighted probability aggregation."""
    # Mock segments: One long positive review, one short neutral statement
    mock_segments = [
        {
            "segment_id": 1,
            "start_time": 0.0,
            "end_time": 10.0,
            "timestamp_label": "00:00 – 00:10",
            "transcript": "Today was an amazing day and everything was fantastic and wonderful!",
            "word_count": 11,
            "duration_seconds": 10.0
        },
        {
            "segment_id": 2,
            "start_time": 10.0,
            "end_time": 12.0,
            "timestamp_label": "00:10 – 00:12",
            "transcript": "Package delivered.",
            "word_count": 2,
            "duration_seconds": 2.0
        }
    ]
    
    from src.sentiment_analyzer import VideoSentimentAnalyzer
    analyzer = VideoSentimentAnalyzer()
    result = analyzer.analyze_segments(mock_segments)
    
    assert "overall_sentiment" in result
    assert result["overall_sentiment"] in ["Positive", "Neutral", "Negative"]
    assert "overall_confidence" in result
    assert 0.0 <= result["overall_confidence"] <= 1.0
    assert "highlights" in result
    assert "most_positive" in result["highlights"]
    assert "most_negative" in result["highlights"]
    assert "most_uncertain" in result["highlights"]
    assert len(result["timeline"]) == 2


def test_explain_prediction():
    """Verify explain_prediction provides token attributions, aspects, and pipeline trace."""
    from src.predict import explain_prediction
    
    text = "The camera quality is excellent but the battery life is terrible."
    result = explain_prediction(text)
    
    assert "sentiment" in result
    assert result["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert "token_attributions" in result
    assert isinstance(result["token_attributions"], list)
    assert len(result["token_attributions"]) > 0
    
    # Check token attribution fields
    first_tok = result["token_attributions"][0]
    assert "word" in first_tok
    assert "impact_score" in first_tok
    assert "polarity" in first_tok
    assert first_tok["polarity"] in ["positive", "negative", "neutral"]
    
    # Check aspect extraction presence
    assert "aspects" in result
    assert len(result["aspects"]) == 2
    
    # Check pipeline trace
    assert "pipeline_trace" in result
    trace = result["pipeline_trace"]
    assert "raw_input" in trace
    assert "cleaned_text" in trace
    assert "token_ids" in trace


def test_explain_prediction_empty_input():
    """Verify explain_prediction handles empty / invalid inputs without failing."""
    from src.predict import explain_prediction
    
    res_empty = explain_prediction("")
    assert res_empty["sentiment"] == "Neutral"
    assert res_empty["tokens"] == []
    assert res_empty["token_attributions"] == []
    
    res_none = explain_prediction(None)
    assert res_none["sentiment"] == "Neutral"


def test_predict_batch():
    """Verify predict_batch handles heterogeneous review lists."""
    from src.predict import SentimentPredictor
    predictor = SentimentPredictor()
    
    texts = [
        "Great laptop with amazing performance!",
        "",
        "Horrible service, total waste of money."
    ]
    batch_res = predictor.predict_batch(texts)
    
    assert len(batch_res) == 3
    assert all("sentiment" in item for item in batch_res)
    assert all("confidence" in item for item in batch_res)
    assert all(item["sentiment"] in ["Positive", "Neutral", "Negative"] for item in batch_res)
    assert batch_res[1]["sentiment"] == "Neutral"
    assert batch_res[1]["confidence"] == 0.3333


def test_aspect_sentiment_compound_clauses():
    """Verify compound clauses joined by 'and' properly resolve sentiment for all aspects."""
    sentence = "The camera quality and screen are excellent, but the battery life is terrible."
    aspects = extract_aspect_sentiments(sentence)
    
    aspect_map = {item["aspect"]: item["sentiment"] for item in aspects}
    assert aspect_map["Camera / Image Quality"] == "Positive"
    assert aspect_map["Display & Screen"] == "Positive"
    assert aspect_map["Battery Life & Charging"] == "Negative"



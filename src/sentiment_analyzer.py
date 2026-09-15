"""
Segment-Level and Video-Level Sentiment Analysis Engine for SentimentVision.
Processes speech segments through the trained LSTM neural network and calculates
duration-weighted overall sentiment, highlights, and timeline trajectories.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    MAX_LENGTH,
    PADDING_TYPE,
    TRUNCATING_TYPE,
    LABEL_TO_SENTIMENT
)
from src.preprocessing import clean_text
from src.model import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VideoSentimentAnalyzer:
    """
    Analyzes timestamped speech segments using the trained LSTM model.
    """
    _model = None
    _tokenizer = None

    def __init__(self):
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load and cache LSTM model and Tokenizer."""
        import tensorflow as tf
        from tensorflow import keras

        if VideoSentimentAnalyzer._model is None or VideoSentimentAnalyzer._tokenizer is None:
            if not LSTM_MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
                raise FileNotFoundError(
                    f"Trained LSTM model or Tokenizer missing at {LSTM_MODEL_PATH}. Run 'python src/train.py' first."
                )
            logger.info(f"Loading SentimentVision LSTM Model from: {LSTM_MODEL_PATH}")
            VideoSentimentAnalyzer._model = keras.models.load_model(LSTM_MODEL_PATH)
            logger.info(f"Loading Tokenizer from: {TOKENIZER_PATH}")
            VideoSentimentAnalyzer._tokenizer = load_tokenizer(TOKENIZER_PATH)
        
        self._model = VideoSentimentAnalyzer._model
        self._tokenizer = VideoSentimentAnalyzer._tokenizer

    def analyze_segments(self, raw_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run LSTM inference on each speech segment and aggregate overall video sentiment.
        
        Args:
            raw_segments: List of segments from SpeechTranscriber.
            
        Returns:
            Dictionary containing:
                - overall_sentiment: str ('Positive', 'Neutral', 'Negative')
                - overall_confidence: float
                - overall_probabilities: Dict[str, float]
                - analyzed_segments: List[Dict[str, Any]]
                - highlights: Dict[str, Any] (most_positive, most_negative, most_uncertain)
                - timeline: List[Dict[str, Any]]
        """
        if not raw_segments:
            return {
                "overall_sentiment": "Neutral",
                "overall_confidence": 0.3333,
                "overall_probabilities": {"negative": 0.3333, "neutral": 0.3334, "positive": 0.3333},
                "analyzed_segments": [],
                "highlights": {},
                "timeline": []
            }

        cleaned_texts = [clean_text(seg["transcript"]) for seg in raw_segments]
        
        # Tokenize and pad
        seqs = self._tokenizer.texts_to_sequences(cleaned_texts)
        padded = self._tokenizer.pad_sequences(
            seqs,
            maxlen=MAX_LENGTH,
            padding=PADDING_TYPE,
            truncating=TRUNCATING_TYPE
        )

        # Predict softmax probabilities for all segments
        probs_all = self._model.predict(padded, verbose=0)

        analyzed_segments = []
        weights = []
        prob_matrix = []

        for i, (seg, cleaned, probs) in enumerate(zip(raw_segments, cleaned_texts, probs_all)):
            pred_class_id = int(np.argmax(probs))
            confidence = float(probs[pred_class_id])
            sentiment_name = LABEL_TO_SENTIMENT.get(pred_class_id, "Neutral")
            
            p_neg = round(float(probs[0]), 4)
            p_neu = round(float(probs[1]), 4)
            p_pos = round(float(probs[2]), 4)

            analyzed_seg = {
                "segment_id": seg.get("segment_id", i + 1),
                "timestamp_start": seg["start_time"],
                "timestamp_end": seg["end_time"],
                "timestamp_label": seg["timestamp_label"],
                "transcript": seg["transcript"],
                "cleaned_text": cleaned,
                "sentiment": sentiment_name,
                "confidence": round(confidence, 4),
                "confidence_pct": f"{confidence * 100:.1f}%",
                "negative_probability": p_neg,
                "neutral_probability": p_neu,
                "positive_probability": p_pos,
                "duration_seconds": seg["duration_seconds"],
                "word_count": seg["word_count"]
            }
            analyzed_segments.append(analyzed_seg)

            # Weight calculation: duration * sqrt(word_count)
            # Prevents tiny 0.2s filler sounds from overpowering substantial statements
            w = max(seg["duration_seconds"], 1.0) * np.sqrt(max(seg["word_count"], 1))
            weights.append(w)
            prob_matrix.append([probs[0], probs[1], probs[2]])

        # ==========================================
        # Weighted Overall Sentiment Aggregation
        # ==========================================
        weights = np.array(weights)
        prob_matrix = np.array(prob_matrix)
        total_weight = np.sum(weights)

        if total_weight > 0:
            weighted_probs = np.sum(prob_matrix * weights[:, np.newaxis], axis=0) / total_weight
        else:
            weighted_probs = np.mean(prob_matrix, axis=0)

        overall_class_id = int(np.argmax(weighted_probs))
        overall_sentiment = LABEL_TO_SENTIMENT.get(overall_class_id, "Neutral")
        overall_confidence = float(weighted_probs[overall_class_id])

        overall_prob_dict = {
            "negative": round(float(weighted_probs[0]), 4),
            "neutral": round(float(weighted_probs[1]), 4),
            "positive": round(float(weighted_probs[2]), 4)
        }

        # ==========================================
        # Key Video Highlights Extraction
        # ==========================================
        most_positive = max(analyzed_segments, key=lambda s: s["positive_probability"])
        most_negative = max(analyzed_segments, key=lambda s: s["negative_probability"])
        most_uncertain = min(analyzed_segments, key=lambda s: s["confidence"])

        highlights = {
            "most_positive": most_positive,
            "most_negative": most_negative,
            "most_uncertain": most_uncertain
        }

        # ==========================================
        # Timeline Data Series
        # ==========================================
        timeline = []
        for seg in analyzed_segments:
            # Midpoint timestamp for line plots
            midpoint = (seg["timestamp_start"] + seg["timestamp_end"]) / 2.0
            
            # Numeric sentiment score: Positive (+1), Neutral (0), Negative (-1) weighted by confidence
            if seg["sentiment"] == "Positive":
                score = seg["confidence"]
            elif seg["sentiment"] == "Negative":
                score = -seg["confidence"]
            else:
                score = 0.0

            timeline.append({
                "timestamp_seconds": round(midpoint, 2),
                "timestamp_label": seg["timestamp_label"],
                "transcript_snippet": seg["transcript"][:50] + ("..." if len(seg["transcript"]) > 50 else ""),
                "sentiment": seg["sentiment"],
                "sentiment_score": round(score, 3),
                "confidence": seg["confidence"],
                "positive_probability": seg["positive_probability"],
                "neutral_probability": seg["neutral_probability"],
                "negative_probability": seg["negative_probability"]
            })

        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": round(overall_confidence, 4),
            "overall_probabilities": overall_prob_dict,
            "analyzed_segments": analyzed_segments,
            "highlights": highlights,
            "timeline": timeline,
            "total_segments_analyzed": len(analyzed_segments)
        }

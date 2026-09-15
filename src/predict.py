"""
Inference and Explainability Engine for Sentiment Analysis.
Provides single, batch, and explainable predictions with word-level attribution,
aspect decomposition, and model comparison.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import pickle
import logging
from typing import Dict, Any, List, Union, Optional
import numpy as np

from src.config import (
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    BASELINE_MODEL_PATH,
    MAX_LENGTH,
    PADDING_TYPE,
    TRUNCATING_TYPE,
    LABEL_TO_SENTIMENT
)
from src.preprocessing import clean_text
from src.model import load_tokenizer, ReviewTokenizer
from src.aspect_analyzer import extract_aspect_sentiments

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Singleton inference and explainability engine that caches the trained LSTM model,
    tokenizer, and baseline model for fast predictions.
    """
    _instance = None
    _model = None
    _tokenizer = None
    _baseline_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentPredictor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Load and cache model, tokenizer, and baseline model."""
        import tensorflow as tf
        from tensorflow import keras

        if not LSTM_MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
            logger.warning("Trained model or tokenizer not found on disk. Initializing in uninitialized state.")
            self._model = None
            self._tokenizer = None
            return

        logger.info(f"Loading SentimentIQ LSTM Model from {LSTM_MODEL_PATH}")
        self._model = keras.models.load_model(LSTM_MODEL_PATH)
        
        logger.info(f"Loading Tokenizer from {TOKENIZER_PATH}")
        self._tokenizer = load_tokenizer(TOKENIZER_PATH)

        if BASELINE_MODEL_PATH.exists():
            try:
                with open(BASELINE_MODEL_PATH, "rb") as f:
                    self._baseline_model = pickle.load(f)
                logger.info(f"Loaded Baseline Model from {BASELINE_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Could not load baseline model: {e}")
                self._baseline_model = None

    def is_ready(self) -> bool:
        """Check if model and tokenizer are successfully loaded."""
        return self._model is not None and self._tokenizer is not None

    def predict(self, text: Union[str, None]) -> Dict[str, Any]:
        """
        Predict sentiment for a single review.
        
        Args:
            text: Raw input review string.
            
        Returns:
            Dictionary with sentiment, confidence, probabilities, and sequence tokens.
        """
        if not self.is_ready():
            self._initialize()
            if not self.is_ready():
                raise RuntimeError("Model and Tokenizer are not trained or loaded. Please run src/train.py first.")

        # Handle empty / invalid input gracefully
        if text is None or not isinstance(text, str) or not text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.3333,
                "probabilities": {
                    "negative": 0.3333,
                    "neutral": 0.3334,
                    "positive": 0.3333
                },
                "cleaned_text": "",
                "tokens": [],
                "token_ids": [],
                "warning": "Input was empty or blank. Defaulting to uniform neutral distribution."
            }

        cleaned = clean_text(text)
        
        # If text became completely empty after removing symbols/HTML
        if not cleaned:
            return {
                "sentiment": "Neutral",
                "confidence": 0.3333,
                "probabilities": {
                    "negative": 0.3333,
                    "neutral": 0.3334,
                    "positive": 0.3333
                },
                "cleaned_text": "",
                "tokens": [],
                "token_ids": [],
                "warning": "Input contained only symbols/special characters with no discernible text."
            }

        # Sequence conversion and padding
        seq = self._tokenizer.texts_to_sequences([cleaned])[0]
        padded = self._tokenizer.pad_sequences([seq], maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

        # Inference
        probs = self._model.predict(padded, verbose=0)[0]
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])

        sentiment_name = LABEL_TO_SENTIMENT.get(pred_class_id, "Neutral")
        tokens = cleaned.split()

        return {
            "sentiment": sentiment_name,
            "confidence": round(confidence, 4),
            "probabilities": {
                "negative": round(float(probs[0]), 4),
                "neutral": round(float(probs[1]), 4),
                "positive": round(float(probs[2]), 4)
            },
            "cleaned_text": cleaned,
            "tokens": tokens,
            "token_ids": seq
        }

    def explain_prediction(self, text: Union[str, None]) -> Dict[str, Any]:
        """
        Produce a comprehensive, explainable sentiment diagnosis for input text.
        Includes word-level attribution/importance, aspect decomposition,
        pipeline transformation steps, and baseline comparison.
        
        Args:
            text: Input string to test and explain.
            
        Returns:
            Dictionary containing detailed analysis, token contributions, aspects, and comparison.
        """
        base_pred = self.predict(text)
        cleaned = base_pred["cleaned_text"]
        words = cleaned.split() if cleaned else []

        # 1. Word-level token attribution (Leave-One-Out perturbation scoring)
        token_attributions = []
        if len(words) > 0 and self.is_ready():
            base_probs = [
                base_pred["probabilities"]["negative"],
                base_pred["probabilities"]["neutral"],
                base_pred["probabilities"]["positive"]
            ]
            pred_sentiment = base_pred["sentiment"]

            # If single word, score directly
            if len(words) == 1:
                w = words[0]
                pol = "positive" if pred_sentiment == "Positive" else ("negative" if pred_sentiment == "Negative" else "neutral")
                token_attributions.append({
                    "word": w,
                    "impact_score": round(base_pred["confidence"], 3),
                    "polarity": pol,
                    "explanation": f"Sole operative keyword carrying {pred_sentiment.lower()} sentiment."
                })
            else:
                # Vectorized leave-one-out ablation for all words in a single batch
                ablated_texts = [" ".join(words[:i] + words[i+1:]) for i in range(len(words))]
                abl_seqs = self._tokenizer.texts_to_sequences(ablated_texts)
                all_abl_padded = self._tokenizer.pad_sequences(
                    abl_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE
                )
                all_abl_probs = self._model.predict(all_abl_padded, verbose=0)

                for i, (w, abl_probs) in enumerate(zip(words, all_abl_probs)):
                    # Positive polarity delta: how much removing this word drops positive probability
                    pos_delta = float(base_probs[2] - abl_probs[2])
                    neg_delta = float(base_probs[0] - abl_probs[0])

                    if pos_delta > 0.03:
                        polarity = "positive"
                        score = round(pos_delta, 3)
                        explanation = "Strong positive driver (increases positive probability)."
                    elif neg_delta > 0.03:
                        polarity = "negative"
                        score = round(neg_delta, 3)
                        explanation = "Strong negative driver (increases negative probability)."
                    elif pos_delta < -0.03:
                        polarity = "negative"
                        score = round(abs(pos_delta), 3)
                        explanation = "Negative anchor (suppresses positive probability)."
                    elif neg_delta < -0.03:
                        polarity = "positive"
                        score = round(abs(neg_delta), 3)
                        explanation = "Positive anchor (suppresses negative probability)."
                    else:
                        polarity = "neutral"
                        score = round(max(abs(pos_delta), abs(neg_delta)), 3)
                        explanation = "Context/Syntactic connector with balanced influence."

                    token_attributions.append({
                        "word": w,
                        "impact_score": score,
                        "polarity": polarity,
                        "pos_delta": round(pos_delta, 4),
                        "neg_delta": round(neg_delta, 4),
                        "explanation": explanation
                    })

        # 2. Aspect-Based Sentiment Analysis
        aspects = extract_aspect_sentiments(text) if text else []

        # 3. Baseline Comparison (TF-IDF + Logistic Regression)
        baseline_pred = None
        if self._baseline_model is not None and cleaned:
            try:
                base_cls = int(self._baseline_model.predict([cleaned])[0])
                base_probs = self._baseline_model.predict_proba([cleaned])[0] if hasattr(self._baseline_model, "predict_proba") else None
                base_sent = LABEL_TO_SENTIMENT.get(base_cls, "Neutral")
                base_conf = float(np.max(base_probs)) if base_probs is not None else 1.0
                baseline_pred = {
                    "model_name": "TF-IDF + Logistic Regression",
                    "sentiment": base_sent,
                    "confidence": round(base_conf, 4),
                    "probabilities": {
                        "negative": round(float(base_probs[0]), 4) if base_probs is not None else 0.0,
                        "neutral": round(float(base_probs[1]), 4) if base_probs is not None else 0.0,
                        "positive": round(float(base_probs[2]), 4) if base_probs is not None else 0.0
                    } if base_probs is not None else {}
                }
            except Exception as e:
                logger.warning(f"Baseline inference failed: {e}")
                baseline_pred = None

        # 4. Pipeline Traceability
        seq_ids = base_pred.get("token_ids", [])
        pipeline_trace = {
            "raw_input": text,
            "cleaned_text": cleaned,
            "word_count": len(words),
            "token_ids": seq_ids,
            "sequence_length": len(seq_ids),
            "max_length_limit": MAX_LENGTH
        }

        return {
            **base_pred,
            "token_attributions": token_attributions,
            "aspects": aspects,
            "baseline_comparison": baseline_pred,
            "pipeline_trace": pipeline_trace
        }

    def predict_batch(self, texts: List[Union[str, None]]) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple reviews simultaneously.
        """
        if not self.is_ready():
            self._initialize()
            if not self.is_ready():
                raise RuntimeError("Model and Tokenizer are not trained or loaded. Please run src/train.py first.")

        cleaned_texts = [clean_text(t) if isinstance(t, str) else "" for t in texts]
        seqs = self._tokenizer.texts_to_sequences(cleaned_texts)
        padded = self._tokenizer.pad_sequences(seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

        probs_all = self._model.predict(padded, verbose=0)
        
        results = []
        for i, (orig_text, cleaned, seq, probs) in enumerate(zip(texts, cleaned_texts, seqs, probs_all)):
            if not cleaned:
                results.append({
                    "review": orig_text,
                    "sentiment": "Neutral",
                    "confidence": 0.3333,
                    "negative_probability": 0.3333,
                    "neutral_probability": 0.3334,
                    "positive_probability": 0.3333
                })
            else:
                pred_class_id = int(np.argmax(probs))
                confidence = float(probs[pred_class_id])
                sentiment_name = LABEL_TO_SENTIMENT.get(pred_class_id, "Neutral")
                results.append({
                    "review": orig_text,
                    "sentiment": sentiment_name,
                    "confidence": round(confidence, 4),
                    "negative_probability": round(float(probs[0]), 4),
                    "neutral_probability": round(float(probs[1]), 4),
                    "positive_probability": round(float(probs[2]), 4)
                })

        return results


def predict_sentiment(text: Union[str, None]) -> Dict[str, Any]:
    """
    Public convenience API for single sentiment prediction.
    """
    predictor = SentimentPredictor()
    return predictor.predict(text)


def explain_prediction(text: Union[str, None]) -> Dict[str, Any]:
    """
    Public convenience API for comprehensive explainable sentiment prediction.
    """
    predictor = SentimentPredictor()
    return predictor.explain_prediction(text)

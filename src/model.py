"""
LSTM Model Architecture, Tokenizer Management, and Serialization Utilities.
"""

import pickle
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReviewTokenizer:
    """
    Robust, production-grade text tokenizer and sequence padder.
    Fitted strictly on training text to prevent data leakage.
    Serializes seamlessly via pickle.
    """
    def __init__(self, num_words: int = 10000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {oov_token: 1}
        self.index_word: Dict[int, str] = {1: oov_token}
        self.word_counts: Dict[str, int] = {}
        self.is_fitted = False

    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from raw training texts."""
        self.word_counts = {}
        for text in texts:
            if not isinstance(text, str):
                continue
            for word in text.strip().split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Reserve index 0 for padding, index 1 for OOV
        self.word_index = {self.oov_token: 1}
        self.index_word = {1: self.oov_token}
        
        for idx, (word, _) in enumerate(sorted_words[: self.num_words - 2], start=2):
            self.word_index[word] = idx
            self.index_word[idx] = word
            
        self.is_fitted = True
        logger.info(f"Tokenizer fitted. Total distinct vocabulary words: {len(self.word_index)}")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to list of integer token ID sequences."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted on training texts before converting to sequences.")
        
        oov_id = self.word_index.get(self.oov_token, 1)
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                sequences.append([])
                continue
            seq = [self.word_index.get(word, oov_id) for word in text.strip().split()]
            sequences.append(seq)
        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        maxlen: int = 150,
        padding: str = "post",
        truncating: str = "post"
    ) -> np.ndarray:
        """
        Pad or truncate integer sequences to fixed maxlen.
        
        Args:
            sequences: List of integer sequence lists.
            maxlen: Maximum length of sequences.
            padding: 'pre' or 'post' padding with zeros.
            truncating: 'pre' or 'post' truncation.
            
        Returns:
            2D NumPy array of shape (num_samples, maxlen).
        """
        num_samples = len(sequences)
        padded_array = np.zeros((num_samples, maxlen), dtype=np.int32)

        for i, seq in enumerate(sequences):
            if not seq:
                continue
            
            # Truncation
            if len(seq) > maxlen:
                if truncating == "pre":
                    trunc_seq = seq[-maxlen:]
                else:
                    trunc_seq = seq[:maxlen]
            else:
                trunc_seq = seq

            # Padding
            if padding == "pre":
                padded_array[i, -len(trunc_seq):] = trunc_seq
            else:
                padded_array[i, :len(trunc_seq)] = trunc_seq

        return padded_array


def build_lstm_model(
    vocab_size: int = 10000,
    embedding_dim: int = 128,
    max_length: int = 150,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    dense_units: int = 64,
    num_classes: int = 3,
    learning_rate: float = 0.001
):
    """
    Construct and compile the production LSTM Deep Learning Model using Keras.
    
    Architecture:
    Input (max_length)
      ↓
    Embedding (vocab_size -> embedding_dim)
      ↓
    LSTM (lstm_units)
      ↓
    Dropout (dropout_rate)
      ↓
    Dense (dense_units, ReLU)
      ↓
    Dense (num_classes, Softmax)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # Fix seed for model weight initialization
    tf.random.set_seed(42)

    inputs = layers.Input(shape=(max_length,), name="input_review_sequence")
    
    # Embedding layer (mask zero for padded positions)
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding"
    )(inputs)
    
    # LSTM layer
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        name="lstm_layer"
    )(x)
    
    # Dropout for regularization against overfitting
    x = layers.Dropout(
        rate=dropout_rate,
        name="dropout_regularization"
    )(x)
    
    # Dense projection layer
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        name="dense_feature_extractor"
    )(x)
    
    # Softmax output layer
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="sentiment_probabilities"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SentimentIQ_LSTM")
    
    # Optimizer & Compilation
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def save_tokenizer(tokenizer: ReviewTokenizer, filepath: Any):
    """Serialize and save tokenizer to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Saved tokenizer to {filepath}")


def load_tokenizer(filepath: Any) -> ReviewTokenizer:
    """Load serialized tokenizer from disk."""
    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

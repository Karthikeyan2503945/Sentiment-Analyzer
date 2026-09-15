import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    RAW_DATASET_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    BASELINE_MODEL_PATH,
    LABEL_MAPPING_PATH,
    TRAINING_HISTORY_PLOT,
    MAX_WORDS,
    MAX_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    NUM_CLASSES,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    LABEL_TO_SENTIMENT,
    PADDING_TYPE,
    TRUNCATING_TYPE
)
from src.data_loader import load_or_generate_dataset, perform_eda, split_and_save_data
from src.model import ReviewTokenizer, build_lstm_model, save_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_training_history(history, save_path):
    """
    Plot and save training & validation loss and accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # Accuracy Plot
    ax1.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", marker="o", color="#2563eb")
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", marker="s", color="#16a34a")
    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # Loss Plot
    ax2.plot(epochs_range, history.history["loss"], label="Training Loss", marker="o", color="#dc2626")
    ax2.plot(epochs_range, history.history["val_loss"], label="Validation Loss", marker="s", color="#d97706")
    ax2.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved training history curves to {save_path}")


def train_baseline_model(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Train a traditional TF-IDF + Logistic Regression baseline model.
    """
    logger.info("Training TF-IDF + Logistic Regression Baseline Model...")
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_WORDS, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"))
    ])
    
    pipeline.fit(train_df["cleaned_review"], train_df["sentiment"])
    val_acc = pipeline.score(val_df["cleaned_review"], val_df["sentiment"])
    logger.info(f"Baseline Model Validation Accuracy: {val_acc:.4f}")
    
    with open(BASELINE_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Saved baseline model to {BASELINE_MODEL_PATH}")
    return pipeline


def run_training():
    """
    Execute full training workflow:
    1. Dataset verification and loading
    2. Tokenizer fitting on train set only
    3. Sequence preparation and padding
    4. LSTM Deep Learning model construction and compilation
    5. Training with callbacks (EarlyStopping, ModelCheckpoint)
    6. Baseline model training
    7. Artifact saving and visualization
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # Ensure dataset is loaded and split
    if not (TRAIN_DATA_PATH.exists() and VAL_DATA_PATH.exists()):
        raw_df = load_or_generate_dataset()
        perform_eda(raw_df)
        train_df, val_df, test_df = split_and_save_data(raw_df)
    else:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)

    logger.info(f"Training dataset size: {len(train_df)} | Validation dataset size: {len(val_df)}")

    # 1. Fit Tokenizer strictly on training text
    logger.info("Fitting Tokenizer on training corpus...")
    tokenizer = ReviewTokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["cleaned_review"].tolist())
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    # 2. Convert texts to padded sequences
    train_seqs = tokenizer.texts_to_sequences(train_df["cleaned_review"].tolist())
    val_seqs = tokenizer.texts_to_sequences(val_df["cleaned_review"].tolist())

    X_train = tokenizer.pad_sequences(train_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    X_val = tokenizer.pad_sequences(val_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    y_train = np.array(train_df["sentiment"], dtype=np.int32)
    y_val = np.array(val_df["sentiment"], dtype=np.int32)

    # 3. Build & compile LSTM model
    logger.info("Building LSTM Neural Network...")
    vocab_size = min(len(tokenizer.word_index) + 2, MAX_WORDS)
    model = build_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        max_length=MAX_LENGTH,
        lstm_units=LSTM_UNITS,
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_UNITS,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )
    model.summary()

    # 4. Training Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode="min"
        ),
        ModelCheckpoint(
            filepath=str(LSTM_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max"
        )
    ]

    # 5. Train LSTM model
    logger.info("Starting LSTM Model Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final best model in keras format
    model.save(LSTM_MODEL_PATH)
    logger.info(f"Model saved successfully to {LSTM_MODEL_PATH}")

    # 6. Save label mapping configuration
    with open(LABEL_MAPPING_PATH, "w") as f:
        json.dump({
            "label_to_sentiment": LABEL_TO_SENTIMENT,
            "max_words": MAX_WORDS,
            "max_length": MAX_LENGTH,
            "vocab_size": vocab_size
        }, f, indent=2)

    # 7. Generate and save training performance curves
    plot_training_history(history, TRAINING_HISTORY_PLOT)

    # 8. Train Baseline Model for comparison
    train_baseline_model(train_df, val_df)
    
    logger.info("Training pipeline completed successfully.")
    return model, history


if __name__ == "__main__":
    run_training()

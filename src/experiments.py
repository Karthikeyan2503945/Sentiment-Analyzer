import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    EXPERIMENTS_CSV_PATH,
    MAX_WORDS,
    MAX_LENGTH,
    BATCH_SIZE,
    RANDOM_SEED,
    PADDING_TYPE,
    TRUNCATING_TYPE
)
from src.model import ReviewTokenizer, build_lstm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_hyperparameter_experiments():
    """
    Run systematic grid experiments and measure actual validation performance.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    if not TRAIN_DATA_PATH.exists() or not VAL_DATA_PATH.exists():
        raise FileNotFoundError("Processed train and validation data must exist before running experiments.")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    tokenizer = ReviewTokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["cleaned_review"].tolist())

    train_seqs = tokenizer.texts_to_sequences(train_df["cleaned_review"].tolist())
    val_seqs = tokenizer.texts_to_sequences(val_df["cleaned_review"].tolist())

    X_train = tokenizer.pad_sequences(train_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    X_val = tokenizer.pad_sequences(val_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    y_train = np.array(train_df["sentiment"], dtype=np.int32)
    y_val = np.array(val_df["sentiment"], dtype=np.int32)

    vocab_size = min(len(tokenizer.word_index) + 2, MAX_WORDS)

    # Controlled parameter sets
    experiment_configs = [
        {"exp_id": 1, "lstm_units": 64,  "embedding_dim": 128, "dropout_rate": 0.5, "dense_units": 64},
        {"exp_id": 2, "lstm_units": 128, "embedding_dim": 128, "dropout_rate": 0.5, "dense_units": 64}, # Baseline Config
        {"exp_id": 3, "lstm_units": 256, "embedding_dim": 128, "dropout_rate": 0.5, "dense_units": 64},
        {"exp_id": 4, "lstm_units": 128, "embedding_dim": 64,  "dropout_rate": 0.5, "dense_units": 64},
        {"exp_id": 5, "lstm_units": 128, "embedding_dim": 256, "dropout_rate": 0.5, "dense_units": 64},
        {"exp_id": 6, "lstm_units": 128, "embedding_dim": 128, "dropout_rate": 0.2, "dense_units": 64},
        {"exp_id": 7, "lstm_units": 128, "embedding_dim": 128, "dropout_rate": 0.7, "dense_units": 64},
    ]

    results = []
    logger.info(f"Starting execution of {len(experiment_configs)} hyperparameter experiments...")

    for cfg in experiment_configs:
        exp_id = cfg["exp_id"]
        units = cfg["lstm_units"]
        emb_dim = cfg["embedding_dim"]
        drop = cfg["dropout_rate"]
        dense = cfg["dense_units"]

        logger.info(f"Running Exp #{exp_id} -> LSTM Units: {units} | Embedding Dim: {emb_dim} | Dropout: {drop}")

        model = build_lstm_model(
            vocab_size=vocab_size,
            embedding_dim=emb_dim,
            max_length=MAX_LENGTH,
            lstm_units=units,
            dropout_rate=drop,
            dense_units=dense,
            num_classes=3,
            learning_rate=0.001
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        start_time = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=6,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0
        )
        elapsed_sec = time.time() - start_time

        val_probs = model.predict(X_val, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        final_train_loss = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])

        results.append({
            "Experiment ID": exp_id,
            "LSTM Units": units,
            "Embedding Dim": emb_dim,
            "Dropout Rate": drop,
            "Dense Units": dense,
            "Val Accuracy": round(val_acc, 4),
            "Val F1-Score": round(val_f1, 4),
            "Train Loss": round(final_train_loss, 4),
            "Val Loss": round(final_val_loss, 4),
            "Training Time (s)": round(elapsed_sec, 2)
        })

    exp_df = pd.DataFrame(results)
    exp_df.to_csv(EXPERIMENTS_CSV_PATH, index=False)
    logger.info(f"Saved experiment results table to {EXPERIMENTS_CSV_PATH}")

    print("\n" + "=" * 80)
    print("                      HYPERPARAMETER EXPERIMENT RESULTS")
    print("=" * 80)
    print(exp_df.to_string(index=False))
    print("=" * 80 + "\n")

    return exp_df


if __name__ == "__main__":
    run_hyperparameter_experiments()

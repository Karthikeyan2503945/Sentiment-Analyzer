import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.config import (
    TEST_DATA_PATH,
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    BASELINE_MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PLOT,
    CLASS_NAMES,
    MAX_LENGTH,
    PADDING_TYPE,
    TRUNCATING_TYPE,
    LABEL_TO_SENTIMENT
)
from src.model import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path):
    """
    Plot and save confusion matrix heatmap with counts and percentages.
    """
    plt.figure(figsize=(8, 6))
    
    # Compute normalized percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Format labels with count and percentage
    annot_matrix = []
    for i in range(len(class_names)):
        row = []
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            row.append(f"{count}\n({pct:.1f}%)")
        annot_matrix.append(row)
        
    sns.heatmap(
        cm,
        annot=np.array(annot_matrix),
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 11, "fontweight": "bold"}
    )
    
    plt.title("SentimentIQ - Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Sentiment Label", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Sentiment Label", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {save_path}")


def analyze_errors(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray):
    """
    Perform qualitative and quantitative error analysis on misclassifications.
    Categorizes errors into Negation, Ambiguity/Mixed, and OOV/Complex syntax.
    """
    error_indices = np.where(y_true != y_pred)[0]
    logger.info(f"Total misclassifications on test set: {len(error_indices)} / {len(y_true)}")
    
    error_samples = []
    for idx in error_indices:
        actual_label = LABEL_TO_SENTIMENT[int(y_true[idx])]
        pred_label = LABEL_TO_SENTIMENT[int(y_pred[idx])]
        conf = float(np.max(probabilities[idx]))
        review_text = str(test_df.iloc[idx]["review"])
        
        # Categorize error type
        lower_text = review_text.lower()
        if any(w in lower_text for w in ["not", "never", "no", "hardly", "barely", "neither", "nor"]):
            category = "Negation Complexity"
        elif any(w in lower_text for w in ["but", "however", "although", "though", "while"]):
            category = "Mixed Sentiment / Contrastive"
        elif actual_label == "Neutral" or pred_label == "Neutral":
            category = "Neutral vs Subtle Sentiment Boundary"
        else:
            category = "Ambiguous Phrasing / Sarcasm"
            
        error_samples.append({
            "review": review_text,
            "actual_sentiment": actual_label,
            "predicted_sentiment": pred_label,
            "confidence": round(conf, 4),
            "error_category": category
        })
        
    return error_samples


def evaluate_models():
    """
    Execute full evaluation suite:
    1. Load test split
    2. Evaluate LSTM model
    3. Evaluate Baseline model
    4. Generate comparative metrics table
    5. Save metrics.json and confusion matrix visualization
    """
    import tensorflow as tf
    from tensorflow import keras

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}. Please run train.py first.")

    test_df = pd.read_csv(TEST_DATA_PATH)
    logger.info(f"Evaluating on {len(test_df)} untouched test samples...")

    # Load Tokenizer and LSTM Model
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    test_seqs = tokenizer.texts_to_sequences(test_df["cleaned_review"].tolist())
    X_test = tokenizer.pad_sequences(test_seqs, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    y_test = np.array(test_df["sentiment"], dtype=np.int32)

    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    lstm_probs = lstm_model.predict(X_test)
    lstm_preds = np.argmax(lstm_probs, axis=1)

    # Calculate LSTM Metrics
    lstm_acc = accuracy_score(y_test, lstm_preds)
    lstm_prec, lstm_rec, lstm_f1, _ = precision_recall_fscore_support(y_test, lstm_preds, average="macro", zero_division=0)
    lstm_prec_w, lstm_rec_w, lstm_f1_w, _ = precision_recall_fscore_support(y_test, lstm_preds, average="weighted", zero_division=0)
    
    # Per-class classification report
    lstm_report = classification_report(y_test, lstm_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    lstm_cm = confusion_matrix(y_test, lstm_preds)

    # Plot Confusion Matrix
    plot_confusion_matrix(lstm_cm, CLASS_NAMES, CONFUSION_MATRIX_PLOT)

    # Baseline Model Evaluation
    baseline_metrics = {}
    if BASELINE_MODEL_PATH.exists():
        with open(BASELINE_MODEL_PATH, "rb") as f:
            baseline_model = pickle.load(f)
        baseline_preds = baseline_model.predict(test_df["cleaned_review"])
        base_acc = accuracy_score(y_test, baseline_preds)
        base_prec, base_rec, base_f1, _ = precision_recall_fscore_support(y_test, baseline_preds, average="macro", zero_division=0)
        base_prec_w, base_rec_w, base_f1_w, _ = precision_recall_fscore_support(y_test, baseline_preds, average="weighted", zero_division=0)
        
        baseline_metrics = {
            "accuracy": float(base_acc),
            "precision_macro": float(base_prec),
            "recall_macro": float(base_rec),
            "f1_macro": float(base_f1),
            "precision_weighted": float(base_prec_w),
            "recall_weighted": float(base_rec_w),
            "f1_weighted": float(base_f1_w),
        }

    # Error Analysis
    error_analysis_results = analyze_errors(test_df, y_test, lstm_preds, lstm_probs)

    # Compile Full Evaluation Report
    results = {
        "dataset": {
            "test_sample_count": len(test_df),
            "class_distribution": {CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(3)}
        },
        "lstm_model": {
            "accuracy": float(lstm_acc),
            "precision_macro": float(lstm_prec),
            "recall_macro": float(lstm_rec),
            "f1_macro": float(lstm_f1),
            "precision_weighted": float(lstm_prec_w),
            "recall_weighted": float(lstm_rec_w),
            "f1_weighted": float(lstm_f1_w),
            "per_class_metrics": lstm_report,
            "confusion_matrix": lstm_cm.tolist()
        },
        "baseline_model": baseline_metrics,
        "comparison_table": [
            {
                "Model": "TF-IDF + Logistic Regression",
                "Accuracy": round(baseline_metrics.get("accuracy", 0.0), 4),
                "Precision (Macro)": round(baseline_metrics.get("precision_macro", 0.0), 4),
                "Recall (Macro)": round(baseline_metrics.get("recall_macro", 0.0), 4),
                "F1-Score (Macro)": round(baseline_metrics.get("f1_macro", 0.0), 4)
            },
            {
                "Model": "LSTM Deep Neural Network",
                "Accuracy": round(float(lstm_acc), 4),
                "Precision (Macro)": round(float(lstm_prec), 4),
                "Recall (Macro)": round(float(lstm_rec), 4),
                "F1-Score (Macro)": round(float(lstm_f1), 4)
            }
        ],
        "error_analysis": error_analysis_results
    }

    # Save to metrics.json
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation metrics and error analysis to {METRICS_PATH}")

    # Print summary table
    print("\n" + "=" * 60)
    print("           MODEL PERFORMANCE COMPARISON (TEST SET)")
    print("=" * 60)
    comp_df = pd.DataFrame(results["comparison_table"])
    print(comp_df.to_string(index=False))
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    evaluate_models()

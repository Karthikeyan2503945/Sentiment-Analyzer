"""
Centralized Configuration Module for SentimentVision.
Manages paths, hyperparameters, video/audio constraints, and model constants.
"""

from pathlib import Path

# ==========================================
# File System Paths
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR, TEMP_DIR, OUTPUTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATASET_PATH = RAW_DATA_DIR / "customer_reviews_3class.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

LSTM_MODEL_PATH = MODELS_DIR / "sentiment_lstm.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"

TRAINING_HISTORY_PLOT = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PLOT = REPORTS_DIR / "confusion_matrix.png"
EXPERIMENTS_CSV_PATH = REPORTS_DIR / "experiments_table.csv"

# ==========================================
# Video & Audio Constraints
# ==========================================
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"]
SUPPORTED_FORMATS = SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS

MAX_VIDEO_SIZE_MB = 100               # Max upload size in Megabytes
MAX_VIDEO_DURATION_SEC = 600          # Max duration: 10 minutes
DEFAULT_WHISPER_MODEL = "tiny"        # Options: tiny, base, small (fast on CPU)
AUDIO_SAMPLE_RATE = 16000             # 16kHz mono standard for Whisper STT

# ==========================================
# Reproducibility & Seed
# ==========================================
RANDOM_SEED = 42

# ==========================================
# Tokenizer & Sequence Hyperparameters
# ==========================================
MAX_WORDS = 10000        # Vocabulary size (top most frequent words)
MAX_LENGTH = 150         # Maximum sequence length for padding/truncation
OOV_TOKEN = "<OOV>"      # Out of Vocabulary token
PADDING_TYPE = "post"    # Sequence padding alignment
TRUNCATING_TYPE = "post" # Sequence truncation alignment

# ==========================================
# LSTM Model Hyperparameters
# ==========================================
EMBEDDING_DIM = 128      # Dimension of dense word embeddings
LSTM_UNITS = 128         # Hidden state size in LSTM layer
DROPOUT_RATE = 0.5       # Dropout probability for regularization
DENSE_UNITS = 64         # Neurons in penultimate dense layer
NUM_CLASSES = 3          # 0: Negative, 1: Neutral, 2: Positive

# ==========================================
# Training Hyperparameters
# ==========================================
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

# ==========================================
# Dataset Splitting
# ==========================================
TRAIN_SPLIT_RATIO = 0.80
VAL_SPLIT_RATIO = 0.10
TEST_SPLIT_RATIO = 0.10

# ==========================================
# Class & Label Mappings
# ==========================================
LABEL_TO_SENTIMENT = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

SENTIMENT_TO_LABEL = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

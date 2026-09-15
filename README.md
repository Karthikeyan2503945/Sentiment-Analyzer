# SentimentAnalyzer — AI-Based Video Sentiment Analysis Using LSTM

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://tensorflow.org)
[![Whisper STT](https://img.shields.io/badge/OpenAI-Whisper-blueviolet.svg)](https://github.com/openai/whisper)
[![MoviePy 2.x](https://img.shields.io/badge/MoviePy-2.1+-purple.svg)](https://zulko.github.io/moviepy/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![Pytest](https://img.shields.io/badge/pytest-32%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Deep Learning and Speech Recognition system that accepts uploaded videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) and audio files (`.wav`, `.mp3`, `.m4a`), extracts high-fidelity 16 kHz audio streams, transcribes timestamped speech using **OpenAI Whisper AI**, and classifies spoken sentiment across time using a **Long Short-Term Memory (LSTM)** Neural Network into **Negative**, **Neutral**, and **Positive** classes with duration-weighted confidence scores.

---

## 📋 Table of Contents

- [Project Overview & Motivation](#-project-overview--motivation)
- [Key Features](#-key-features)
- [Two-Stage AI System Architecture](#-two-stage-ai-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Mathematical Framework: Duration-Weighted Sentiment Aggregation](#-mathematical-framework-duration-weighted-sentiment-aggregation)
- [Interactive Workspaces & Streamlit Dashboard](#-interactive-workspaces--streamlit-dashboard)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Model Training, Evaluation & Experiments](#-model-training-evaluation--experiments)
- [Empirical Model Benchmarks](#-empirical-model-benchmarks)
- [Automated Pytest Suite](#-automated-pytest-suite)
- [Edge Cases & Error Handling](#-edge-cases--error-handling)
- [Resolved Bugs & Technical Improvements](#-resolved-bugs--technical-improvements)
- [License](#-license)

---

## 🌟 Project Overview & Motivation

Modern customer reviews, video feedback, product unboxings, and user interviews express sentiment dynamically over time. Traditional text-only classifiers require users to manually transcribe speech or copy-paste text strings.

**SentimentVision** bridges the gap between multimodal video input and sequential NLP intelligence by orchestrating:
1. **Speech Recognition Stage (Whisper):** Converts spoken audio into clean, timestamped text intervals with start and end timestamps.
2. **Sequential Deep Learning Stage (LSTM):** Evaluates sequential language patterns, negations, and context to produce granular class probabilities and dynamic sentiment trajectory timelines.
3. **Interactive Explainability Studio:** Provides token-level attribution (word influence scores), Aspect-Based Sentiment Analysis (ABSA), and head-to-head model comparisons against a TF-IDF + Logistic Regression baseline.

---

## ✨ Key Features

- **Multi-Format Video & Audio Ingestion:** Validates and processes `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.wav`, `.mp3`, and `.m4a` files with file size checks (up to 100 MB) and duration constraints (up to 10 minutes).
- **Cross-Version Audio Extraction:** Built on MoviePy 2.x and 1.x compatible pipelines, extracting 16 kHz mono PCM WAV streams with automatic resource closing.
- **Timestamped Speech-to-Text (STT):** Uses Whisper AI to automatically detect language and segment speech into precise intervals (e.g., `00:00 – 00:06`).
- **Segment-Level LSTM Sentiment Classifier:** Predicts 3 classes (`Negative`, `Neutral`, `Positive`) with calibrated softmax confidence scores.
- **Duration-Weighted Overall Sentiment Aggregation:** Aggregates video sentiment using duration and word-count weighting rather than unweighted sentence averaging.
- **Interactive Sentiment Timeline:** Visualizes sentiment polarity transitions throughout the entire duration of the video.
- **Automated Video Highlights Extraction:**
  - 🌟 **Most Positive Segment**
  - ⚠️ **Most Negative Segment**
  - ❓ **Most Uncertain Segment** (lowest maximum softmax probability)
- **Interactive Explainability Studio:**
  - **Word Attribution Chips & Bar Charts:** Vectorized leave-one-out perturbation scoring to highlight positive, negative, and neutral driver words.
  - **Aspect-Based Sentiment Decomposition (ABSA):** Resolves sentiment per entity (Camera, Battery, Display, Sound, Performance, Customer Support, etc.) with proximity matching and negation detection.
  - **Side-by-Side Model Comparison:** Real-time inference comparison between the LSTM Neural Network and the TF-IDF + Logistic Regression baseline.
  - **Preprocessing Traceability Inspector:** Inspects raw strings, normalized tokens, vocabulary IDs, and sequence padding shapes.
- **Batch Evaluation & CSV Stress-Testing:** Upload CSV files with review columns or paste multiple lines for throughput testing and class distribution metrics.
- **Persistent State Management:** Streamlit UI retains analysis results across tab changes and report downloads (`sentiment_analysis.csv` and `transcript.txt`).
- **Pre-loaded Demo Scenarios:** Instant one-click evaluation on positive, negative, neutral, and mixed review scenarios.

---

## 🧩 Two-Stage AI System Architecture

```
                       Uploaded Video File (.mp4, .mov, .avi, .mkv)
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │     Stage 1: Video & Audio Pipeline     │
                       │  - Format & Size Validation             │
                       │  - 16 kHz Mono Audio Extraction (WAV)   │
                       │  - Whisper Speech Recognition           │
                       │  - Timestamp Segmentation (MM:SS)       │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                                Timestamped Transcript Segments
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │   Stage 2: LSTM Sentiment Intelligence  │
                       │  - Negation-Preserving Text Cleaner     │
                       │  - Tokenization (Vocab: 10,000)         │
                       │  - Sequence Padding (Max: 150)          │
                       │  - Word Embedding (128-dim)             │
                       │  - LSTM Layer (128 units + Dropout 0.5) │
                       │  - Dense ReLU & Softmax (3 Classes)     │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │    Overall Aggregation & Analytics      │
                       │  - Duration-Weighted Overall Sentiment  │
                       │  - Dynamic Sentiment Timeline Chart     │
                       │  - Key Highlights (Pos/Neg/Uncertain)   │
                       │  - CSV & TXT Export                     │
                       └─────────────────────────────────────────┘
```

> **Note on Component Separation:** Whisper STT and the LSTM Sentiment Classifier are distinct AI modules. Whisper extracts words with timestamps from acoustic waveforms, while the LSTM model classifies the linguistic sentiment of those word sequences.

---

## 📁 Project Directory Structure

```
Sentiment Analysis/
│
├── data/
│   ├── raw/
│   │   └── customer_reviews_3class.csv  # 3-class review benchmark dataset
│   └── processed/
│       ├── train.csv                    # 80% Stratified Training split
│       ├── val.csv                      # 10% Stratified Validation split
│       └── test.csv                     # 10% Stratified Test split
│
├── models/
│   ├── sentiment_lstm.keras             # Trained Keras LSTM deep learning model
│   ├── tokenizer.pkl                    # Serialized ReviewTokenizer
│   ├── baseline_model.pkl               # TF-IDF + Logistic Regression baseline
│   ├── label_mapping.json               # Class and ID metadata
│   └── metrics.json                     # Empirical test evaluation metrics & error audit
│
├── notebooks/
│   └── sentiment_analysis.ipynb         # Interactive Jupyter Notebook walkthrough
│
├── reports/
│   ├── training_history.png             # Loss & Accuracy learning curves
│   ├── confusion_matrix.png             # Test set confusion matrix
│   └── experiments_table.csv            # Controlled hyperparameter search table
│
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Centralized configurations and constraints
│   ├── data_loader.py                   # Ingestion, EDA, and zero-leakage splitting
│   ├── preprocessing.py                 # Negation-preserving text normalizer
│   ├── model.py                         # LSTM architecture and ReviewTokenizer class
│   ├── train.py                         # Model training pipeline with callbacks
│   ├── evaluate.py                      # Test set evaluation and error analysis
│   ├── experiments.py                   # Controlled hyperparameter search runner
│   ├── speech_to_text.py                # Whisper STT and timestamp extraction
│   ├── video_processor.py               # Video/audio validation & cross-version MoviePy extraction
│   ├── sentiment_analyzer.py            # Segment-level & duration-weighted aggregation
│   ├── aspect_analyzer.py               # Aspect-Based Sentiment Analysis (ABSA) & clause parsing
│   ├── predict.py                       # Singleton inference & vectorized explainability engine
│   └── predictor.py                     # Unified video analysis orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py            # Preprocessing & negation unit tests (9 tests)
│   ├── test_video.py                    # Video validation, audio extraction & cleanup (8 tests)
│   ├── test_prediction.py               # Segment aggregation, explainability & ABSA (10 tests)
│   └── test_csv_validation.py           # Batch CSV upload validation tests (5 tests)
│
├── temp/                                # Temporary audio extraction cache
├── outputs/                             # Exported reports directory
├── app.py                               # Production Streamlit Web Product
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Comprehensive documentation
├── .gitignore                           # Git ignore rules
└── LICENSE                              # MIT License
```

---

## ⏱️ Mathematical Framework: Duration-Weighted Sentiment Aggregation

Rather than treating a brief 0.5-second filler utterance equally with a 15-second detailed evaluation, SentimentVision weights each segment $i$ by its duration and word count:

$$w_i = \max(\text{duration}_i, 1.0) \times \sqrt{\max(\text{word\_count}_i, 1)}$$

The overall video class probabilities $\bar{P}_{\text{class}}$ are computed as:

$$\bar{P}_{\text{class}} = \frac{\sum_{i=1}^N w_i \cdot P_{i, \text{class}}}{\sum_{i=1}^N w_i}$$

$$\text{Overall Sentiment} = \underset{\text{class} \in \{\text{Negative, Neutral, Positive}\}}{\operatorname{argmax}} \bar{P}_{\text{class}}$$

This ensures that substantive speech segments drive the overall assessment while short interjections do not disproportionately skew the classification.

---

## 🖥️ Interactive Workspaces & Streamlit Dashboard

The Streamlit dashboard (`app.py`) is organized into four purpose-built workspaces:

1. **🎬 Video Sentiment Analysis:**
   - Upload any video or audio file with instant preview and format validation.
   - Run one-click demo scenarios (Positive Review, Negative Complaint, Neutral Overview, Mixed Timeline).
   - View overall video sentiment badge, confidence gauge, and probability distribution.
   - Inspect key highlights (Most Positive, Most Negative, Most Uncertain segment).
   - Interactive Plotly sentiment trajectory timeline across video timestamps.
   - Full segment-by-segment table with timestamps, transcripts, and probability breakdown.
   - Download reports in CSV (`sentiment_analysis.csv`) and TXT (`transcript.txt`).
   - Results persist across downloads and reruns with a dedicated clear button.

2. **🔬 Input Testing & Explainability Studio:**
   - **Single Sentence Diagnosis:** Test custom sentences or select linguistic challenge presets (Negation challenges, Strong Negation, Mixed Contrastive, Factual Neutral, Sarcasm, High Praise).
   - **Token-Level Attribution:** Vectorized leave-one-out word perturbation scoring displays driver words with color-coded chips and impact bar charts.
   - **Aspect-Based Sentiment Decomposition (ABSA):** Detects entity categories (Camera, Battery, Display, Sound, Performance, Build Quality, Support, Delivery, Value, Comfort) and resolves sentiment per clause with negation detection.
   - **Side-by-Side Model Comparison:** Compares LSTM predictions with TF-IDF + Logistic Regression baseline on the exact same input.
   - **Step-by-Step Preprocessing Trace:** View raw input, normalized string, tokens, vocabulary IDs, and sequence tensor shapes.
   - **Batch Evaluation & CSV Stress-Testing:** Upload CSV or paste multiple reviews to evaluate class distributions and export results.

3. **📊 Model Benchmarks & Error Analysis:**
   - Head-to-head empirical comparison table on the untouched test split.
   - Confusion matrix heatmap with count and percentage annotations.
   - Training loss and accuracy learning curves.
   - Controlled hyperparameter search results across LSTM units, embedding dimensions, and dropout rates.
   - Qualitative error analysis audit categorizing misclassifications by error category (Negation Complexity, Mixed/Contrastive, Neutral Boundary, Ambiguity).

4. **🧠 Architecture & Deep Learning Guide:**
   - Full pipeline architecture flow and neural network layer specifications.
   - Deep learning NLP concepts and mathematical formulation.

---

## 💻 Installation & Environment Setup


### Install Dependencies
```bash
pip install -r requirements.txt
```

> **FFmpeg Note:** Video and audio processing requires FFmpeg. It is automatically bundled via `imageio-ffmpeg` in `requirements.txt`. For system-level FFmpeg, ensure `ffmpeg` is accessible in your system PATH.

---

## 🚀 Model Training, Evaluation & Experiments

Execute the full machine learning lifecycle from scratch:

```bash
# 1. Prepare benchmark dataset, run EDA, and generate zero-leakage stratified splits
python src/data_loader.py

# 2. Train LSTM Deep Neural Network and Baseline model
python src/train.py

# 3. Evaluate models on untouched test split and generate metrics.json & confusion matrix
python src/evaluate.py

# 4. Run controlled hyperparameter grid search
python src/experiments.py
```

*Note: On Windows systems with the Python launcher, prefix commands with `py -3.13` (e.g., `py -3.13 src/train.py`).*

---

## 📊 Empirical Model Benchmarks

Evaluation performed on an untouched stratified test split (10% hold-out):

| Model Architecture | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + Logistic Regression** | **86.00%** | **88.26%** | **85.78%** | **85.91%** | Fast n-gram baseline, ignores deep sequence context |
| **SentimentVision LSTM Network** | **86.00%** | **86.55%** | **85.78%** | **85.67%** | Captures sequential word order, negations, and dependencies |

### Hyperparameter Search Summary
- **Embedding Dimension:** 128 (optimal balance of semantic capacity and generalization)
- **LSTM Units:** 128 (hidden units with Dropout 0.5)
- **Vocabulary Size:** 10,000 words (with `<OOV>` token for unseen words)
- **Sequence Length:** 150 tokens (post-padded and post-truncated)
- **Optimizer:** Adam ($\text{learning\_rate} = 0.001$, early stopping patience 3)

---

## 🧪 Automated Pytest Suite

Run all 32 automated unit tests covering video processing, audio extraction, segmentation, weighted aggregation, negation preservation, explainability, and batch CSV validation:

```powershell
pytest -v
```

### Test Suite Breakdown:
- **`tests/test_video.py` (8 tests):** Video formats, audio formats, unsupported file rejection, empty file rejection, oversized file checks, temp file cleanup, audio extraction via MoviePy, and missing file error handling.
- **`tests/test_prediction.py` (10 tests):** Tokenizer OOV and padding, aspect sentiment extraction, negation in aspect extraction, compound clauses joined by 'and', response contract schema, timestamp formatting, weighted segment aggregation, vectorized explainability attribution, empty input resilience, and batch prediction.
- **`tests/test_preprocessing.py` (9 tests):** Lowercase normalization, HTML tag stripping, URL/email removal, negation preservation ("not", "never", contractions), empty/null input resilience, punctuation cleaning, long review handling, batch preprocessing, and token extraction.
- **`tests/test_csv_validation.py` (5 tests):** Valid CSV parsing, case-insensitive column headers, missing column handling, empty CSV handling, and empty review row filtering.

**Result:** `32 passed in ~20s (100% pass rate)`.

---

## ⚠️ Edge Cases & Error Handling

- **Videos with No Audio Track:** Detected during validation and raises: *"No audio track was detected in this video file. SentimentVision requires spoken audio."*
- **Silent or Music-Only Videos:** Whisper and segment verification detects absence of speech and raises: *"No speech could be detected in the audio track. The video may contain only silence or music."*
- **Oversized / Over-length Files:** Enforces `MAX_VIDEO_SIZE_MB = 100 MB` and `MAX_VIDEO_DURATION_SEC = 600 s` (10 minutes).
- **Temporary File Lifecycle:** Automatically cleans up extracted WAV tracks and temporary upload files in `temp/` upon completion.
- **Empty or Malformed Text Inputs:** Handled gracefully with fallback neutral uniform distributions rather than throwing unhandled exceptions.

---

## 🔧 Resolved Bugs & Technical Improvements

This version incorporates comprehensive stability and performance fixes:

1. **MoviePy 2.x Import Compatibility Fix (`src/video_processor.py`):**
   - *Issue:* MoviePy v2.x deprecated and removed `moviepy.editor`, causing `ModuleNotFoundError` during video and audio extraction.
   - *Fix:* Replaced deprecated imports with top-level imports (`from moviepy import VideoFileClip, AudioFileClip`) with fallback support for MoviePy 1.x.
2. **Streamlit Session State Demo Scenario Persistence (`app.py`):**
   - *Issue:* Demo button clicks were lost on subsequent Streamlit rerun cycles when clicking "Analyze Spoken Sentiment".
   - *Fix:* Stored scenario selection in `st.session_state["selected_scenario"]`, enabling reliable demo analysis.
3. **Analysis Dashboard Persistence Across Downloads (`app.py`):**
   - *Issue:* Clicking CSV or TXT report download buttons triggered a Streamlit rerun that reset the analysis view.
   - *Fix:* Cached analysis results in `st.session_state["video_analysis_result"]`, keeping charts, tables, and downloads visible until explicitly cleared.
4. **Explainability Studio Text Area Preset Synchronization (`app.py`):**
   - *Issue:* Selecting linguistic challenge presets updated an internal variable but left the Streamlit text area stuck on old text.
   - *Fix:* Synchronized widget key `test_sentence_box` directly with preset selections.
5. **Aspect-Based Sentiment Analysis on Compound Clauses (`src/aspect_analyzer.py`):**
   - *Issue:* Splitting on the word "and" split compound subjects like "The camera quality and screen are excellent", falsely classifying camera as Neutral.
   - *Fix:* Refined clause boundaries to split along punctuation and contrastive conjunctions, using proximity-based sentiment resolution for compound entities.
6. **Vectorized Leave-One-Out Attribution Scoring (`src/predict.py`):**
   - *Issue:* Explainability ran individual `model.predict(...)` calls in a sequential loop per word (taking 2–4 seconds).
   - *Fix:* Vectorized ablated sentences into a single batch prediction call, reducing attribution latency to under 100 ms.
7. **Class-Level Model Caching (`src/sentiment_analyzer.py`):**
   - *Issue:* `VideoSentimentAnalyzer` reloaded Keras model weights from disk on every instantiation.
   - *Fix:* Implemented class-level caching on `VideoSentimentAnalyzer._model` and `_tokenizer`.
8. **Automated Audio Extraction Test Coverage (`tests/test_video.py`):**
   - *Issue:* No unit test existed for `extract_audio_from_video`, allowing MoviePy import regressions to pass silently.
   - *Fix:* Added tests verifying audio extraction and missing file error handling.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

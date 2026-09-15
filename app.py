"""
Sentiment Analyzer — AI-Based Video & Text Sentiment Analysis Using LSTM & Speech Recognition.
Production Streamlit Application with Interactive Input Testing & Explainability Suite.
"""

import os
import io
import json
import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set page config FIRST
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    BASELINE_MODEL_PATH,
    METRICS_PATH,
    EXPERIMENTS_CSV_PATH,
    CONFUSION_MATRIX_PLOT,
    TRAINING_HISTORY_PLOT,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    MAX_VIDEO_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    DEFAULT_WHISPER_MODEL,
    MAX_WORDS,
    MAX_LENGTH,
    EMBEDDING_DIM,
    LSTM_UNITS,
    DROPOUT_RATE,
    DENSE_UNITS,
    TEMP_DIR
)
from src.video_processor import validate_media_file, extract_audio_from_video, cleanup_temp_file
from src.speech_to_text import transcribe_audio, format_segment_interval
from src.sentiment_analyzer import VideoSentimentAnalyzer
from src.predict import predict_sentiment, explain_prediction, SentimentPredictor
from src.aspect_analyzer import extract_aspect_sentiments


# Custom CSS for Classic White & Golden Theme with Roboto Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,500;0,700;0,900;1,400;1,500&family=Roboto+Mono:wght@400;500;600;700&display=swap');

    /* Global Roboto Font & Classic White Canvas */
    html, body, [class*="css"], .stApp {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background-color: #ffffff;
        color: #1c1d21;
    }

    /* Main Golden Gradient Header */
    .main-header {
        font-family: 'Roboto', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, #996515 0%, #c59b27 30%, #dfb743 60%, #b8860b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.0rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 10px rgba(212, 175, 55, 0.15);
    }
    .sub-header {
        font-family: 'Roboto', sans-serif;
        color: #786227;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
        letter-spacing: -0.005em;
    }
    
    /* Classic White & Gold Hero Container */
    .hero-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.8rem;
        border: 1px solid #e8dfc8;
        box-shadow: 0 8px 24px -4px rgba(212, 175, 55, 0.12), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    .highlight-card {
        background: #fdfcf7;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #d4af37;
        border-top: 1px solid #f2e9d2;
        border-right: 1px solid #f2e9d2;
        border-bottom: 1px solid #f2e9d2;
        box-shadow: 0 4px 12px -2px rgba(212, 175, 55, 0.08);
        margin-bottom: 1rem;
    }

    /* Golden & Sentiment Badges */
    .sentiment-positive {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        color: #065f46;
        border: 1px solid #86efac;
        border-left: 5px solid #10b981;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: 800;
        font-family: 'Roboto', sans-serif;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.1);
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #fffdf0 0%, #ffffff 100%);
        color: #92400e;
        border: 1px solid #fde047;
        border-left: 5px solid #d4af37;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: 800;
        font-family: 'Roboto', sans-serif;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 2px 8px rgba(212, 175, 55, 0.12);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
        color: #991b1b;
        border: 1px solid #fca5a5;
        border-left: 5px solid #ef4444;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        font-weight: 800;
        font-family: 'Roboto', sans-serif;
        display: inline-block;
        font-size: 1.3rem;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1);
    }
    
    /* Roboto Mono Token Chips */
    .word-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px;
        border-radius: 8px;
        font-size: 0.92rem;
        font-weight: 600;
        font-family: 'Roboto Mono', monospace;
        transition: all 0.18s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .word-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(212, 175, 55, 0.2);
    }
    .word-pos {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .word-neg {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .word-neu {
        background-color: #faf8f2;
        color: #6b5c35;
        border: 1px solid #e4d7b5;
    }

    /* Aspect Badges */
    .aspect-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 3px;
        font-weight: 700;
        font-size: 0.88rem;
        font-family: 'Roboto', sans-serif;
    }
    .aspect-pos { background-color: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .aspect-neg { background-color: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .aspect-neu { background-color: #fffdf0; color: #b45309; border: 1px solid #fde68a; }

    /* Streamlit Buttons: Classic White & Luxury Gold */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #dfb743 0%, #c59b27 50%, #b8860b 100%) !important;
        color: #ffffff !important;
        border: 1px solid #a67c1e !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(197, 155, 39, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #ebd169 0%, #dfb743 50%, #c59b27 100%) !important;
        box-shadow: 0 6px 20px rgba(197, 155, 39, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #786227 !important;
        border: 1px solid #e0d3b6 !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #faf8f2 !important;
        border-color: #c59b27 !important;
        color: #574415 !important;
        box-shadow: 0 2px 8px rgba(197, 155, 39, 0.15) !important;
    }

    /* Streamlit Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #fbf9f4 !important;
        border-right: 1px solid #eae0cb !important;
        font-family: 'Roboto', sans-serif !important;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Roboto', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #faf8f2 !important;
        color: #996515 !important;
        border-bottom: 2px solid #c59b27 !important;
    }

    /* Hide Streamlit Top-Right Buttons (Deploy Button, Menu, Toolbar, Status Widget) */
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    header {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    footer {
        display: none !important;
        visibility: hidden !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer():
    """Cache the LSTM sentiment analyzer."""
    return VideoSentimentAnalyzer()


def load_metrics_data():
    """Load cached metrics if available."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# Built-in demo audio scenarios for immediate testing
SAMPLE_SCENARIOS = {
    "🌟 Positive Product Review": [
        {"start_time": 0.0, "end_time": 6.5, "timestamp_label": "00:00 – 00:06", "transcript": "Today I am reviewing this laptop and the build quality is absolutely fantastic.", "duration_seconds": 6.5, "word_count": 14},
        {"start_time": 6.5, "end_time": 14.0, "timestamp_label": "00:06 – 00:14", "transcript": "The battery life lasts over 14 hours and the screen resolution is stunning.", "duration_seconds": 7.5, "word_count": 13},
        {"start_time": 14.0, "end_time": 20.0, "timestamp_label": "00:14 – 00:20", "transcript": "Customer support answered my setup questions immediately. Highly recommended!", "duration_seconds": 6.0, "word_count": 9}
    ],
    "⚠️ Negative Customer Experience": [
        {"start_time": 0.0, "end_time": 5.0, "timestamp_label": "00:00 – 00:05", "transcript": "I am deeply disappointed with this order.", "duration_seconds": 5.0, "word_count": 7},
        {"start_time": 5.0, "end_time": 12.0, "timestamp_label": "00:05 – 00:12", "transcript": "The unit stopped working completely after two days and the screen started flickering.", "duration_seconds": 7.0, "word_count": 13},
        {"start_time": 12.0, "end_time": 18.0, "timestamp_label": "00:12 – 00:18", "transcript": "Customer service was extremely rude and refused to honor their refund policy.", "duration_seconds": 6.0, "word_count": 12}
    ],
    "⚖️ Neutral Technical Overview": [
        {"start_time": 0.0, "end_time": 7.0, "timestamp_label": "00:00 – 00:07", "transcript": "The device measures 10 inches by 5 inches and weighs 300 grams.", "duration_seconds": 7.0, "word_count": 12},
        {"start_time": 7.0, "end_time": 14.0, "timestamp_label": "00:07 – 00:14", "transcript": "It includes two USB-C ports on the left side and a power toggle on the rear.", "duration_seconds": 7.0, "word_count": 16},
        {"start_time": 14.0, "end_time": 20.0, "timestamp_label": "00:14 – 00:20", "transcript": "The package arrived in standard brown cardboard with a one year warranty.", "duration_seconds": 6.0, "word_count": 12}
    ],
    "🔄 Dynamic Mixed Timeline": [
        {"start_time": 0.0, "end_time": 6.0, "timestamp_label": "00:00 – 00:06", "transcript": "Welcome back everyone, today we test the new smart speaker which has superb sound quality.", "duration_seconds": 6.0, "word_count": 15},
        {"start_time": 6.0, "end_time": 12.0, "timestamp_label": "00:06 – 00:12", "transcript": "The device connects over standard Wi-Fi and Bluetooth version 5.2.", "duration_seconds": 6.0, "word_count": 10},
        {"start_time": 12.0, "end_time": 19.0, "timestamp_label": "00:12 – 00:19", "transcript": "However the mobile application crashes constantly and the setup guide is terrible.", "duration_seconds": 7.0, "word_count": 12},
        {"start_time": 19.0, "end_time": 26.0, "timestamp_label": "00:19 – 00:26", "transcript": "Overall it is not bad at all for the price, but software definitely needs improvement.", "duration_seconds": 7.0, "word_count": 14}
    ]
}

# Linguistic Challenge Test Cases for Explainability Studio
TEST_PRESETS = {
    "⚡ Negation Challenge ('Not bad at all')": "I do not dislike this phone, it is actually not bad at all.",
    "🚫 Strong Negation ('Never buy')": "Never buy this product, it cannot perform properly and customer support was not helpful.",
    "⚖️ Mixed / Contrastive ('Good X but bad Y')": "The camera quality and screen are excellent, but the battery life is terrible and crashes often.",
    "📦 Subtle Neutral ('Factual Specifications')": "The package measures 10 inches by 5 inches and weighs 300 grams with two USB ports.",
    "🎭 Sarcasm / Irony ('Oh wonderful')": "Oh wonderful, another software update that completely broke my wireless connection.",
    "🌟 High Praise ('Flawless Experience')": "Absolutely fantastic product! Exceeded all my expectations with outstanding build quality."
}


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluent/96/video-call.png", width=64)
        st.title("Sentiment Analyzer")
        st.caption("AI Video & Text Sentiment Intelligence")
        st.markdown("---")

        st.markdown("### ⚙️ Video & Audio Limits")
        st.markdown(f"""
        - **Formats:** `{', '.join(SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS)}`
        - **Max File Size:** `{MAX_VIDEO_SIZE_MB} MB`
        - **Max Duration:** `{MAX_VIDEO_DURATION_SEC // 60} mins`
        - **Audio Sample Rate:** `16 kHz Mono`
        """)

        st.markdown("---")
        st.markdown("### 🧩 Two-Stage AI Architecture")
        st.info("""
        **1. Speech Recognition (Whisper AI):**
        Extracts timestamped speech transcript from video audio track.
        
        **2. Sentiment Classification (LSTM):**
        Processes tokenized speech sequences through LSTM Recurrent Neural Network.
        """)

        st.markdown("---")
        st.markdown("### 🧠 LSTM Model Parameters")
        st.markdown(f"""
        - **Vocab Size:** `{MAX_WORDS:,}`
        - **Embedding Dim:** `{EMBEDDING_DIM}`
        - **LSTM Units:** `{LSTM_UNITS}`
        - **Dropout:** `{DROPOUT_RATE}`
        - **Classes:** `Negative (0), Neutral (1), Positive (2)`
        """)

    # Main Header
    st.markdown('<div class="main-header">Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Video & Text Sentiment Analysis with Interactive Explainability Studio</div>', unsafe_allow_html=True)

    # Tabs
    tab_video, tab_test_suite, tab_metrics, tab_arch = st.tabs([
        "🎬 Video Sentiment Analysis",
        "🔬 Input Testing & Explainability Studio",
        "📊 Model Benchmarks & Error Analysis",
        "🧠 Architecture & Deep Learning Guide"
    ])

    # ==========================================
    # TAB 1: Video Sentiment Analysis
    # ==========================================
    with tab_video:
        st.subheader("Analyze Spoken Sentiment in Customer & Product Videos")
        st.write("Upload a video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) or select a pre-recorded demo scenario to analyze speech sentiment across time.")

        # Preset Sample Selector
        # Preset Sample Selector
        st.markdown("**Quick Demo Scenarios:**")
        if "selected_scenario" not in st.session_state:
            st.session_state["selected_scenario"] = None

        cols_demo = st.columns(4)
        if cols_demo[0].button("🌟 Positive Review", use_container_width=True):
            st.session_state["selected_scenario"] = "🌟 Positive Product Review"
            st.session_state["video_analysis_result"] = None
        if cols_demo[1].button("⚠️ Negative Complaint", use_container_width=True):
            st.session_state["selected_scenario"] = "⚠️ Negative Customer Experience"
            st.session_state["video_analysis_result"] = None
        if cols_demo[2].button("⚖️ Neutral Overview", use_container_width=True):
            st.session_state["selected_scenario"] = "⚖️ Neutral Technical Overview"
            st.session_state["video_analysis_result"] = None
        if cols_demo[3].button("🔄 Mixed Timeline", use_container_width=True):
            st.session_state["selected_scenario"] = "🔄 Dynamic Mixed Timeline"
            st.session_state["video_analysis_result"] = None

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload Video File:",
            type=[fmt.replace(".", "") for fmt in (SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS)],
            help="Upload MP4, MOV, AVI, MKV, WEBM video or WAV, MP3 audio."
        )

        video_path = None
        raw_segments = None

        if uploaded_file is not None:
            # When user uploads file, clear demo scenario selection
            st.session_state["selected_scenario"] = None
            file_suffix = Path(uploaded_file.name).suffix.lower()
            temp_input_path = TEMP_DIR / f"upload_{int(time.time())}{file_suffix}"
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = temp_input_path

            # Preview
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                st.markdown("#### 📽️ Video Preview")
                if file_suffix in SUPPORTED_VIDEO_FORMATS:
                    st.video(str(temp_input_path))
                else:
                    st.audio(str(temp_input_path))

            with col_v2:
                st.markdown("#### 📁 File Information")
                st.markdown(f"""
                - **File Name:** `{uploaded_file.name}`
                - **File Size:** `{uploaded_file.size / (1024*1024):.2f} MB`
                - **Format:** `{file_suffix}`
                """)

        elif st.session_state.get("selected_scenario") is not None:
            scenario_name = st.session_state["selected_scenario"]
            st.info(f"Loaded Demo Scenario: **{scenario_name}**")
            raw_segments = SAMPLE_SCENARIOS[scenario_name]

        # Action Buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, col_clear = st.columns([1, 1])
        with col_btn:
            analyze_clicked = st.button("🚀 Analyze Spoken Sentiment", type="primary", use_container_width=True)
        with col_clear:
            if st.session_state.get("video_analysis_result") is not None or st.session_state.get("selected_scenario") is not None:
                if st.button("🗑️ Clear Results / Reset", use_container_width=True):
                    st.session_state["video_analysis_result"] = None
                    st.session_state["selected_scenario"] = None
                    st.session_state["video_analysis_duration"] = None
                    st.rerun()

        if analyze_clicked:
            if video_path is None and raw_segments is None:
                st.warning("Please upload a video or select a demo scenario before analyzing.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    analyzer = get_analyzer()

                    if video_path is not None:
                        # 1. Validation
                        status_text.text("1/4: Validating video format and size constraints...")
                        progress_bar.progress(20)
                        is_valid, err_msg, media_type = validate_media_file(video_path)
                        if not is_valid:
                            st.error(f"❌ {err_msg}")
                            st.stop()

                        # 2. Extract Audio
                        status_text.text("2/4: Extracting audio stream (16kHz mono WAV)...")
                        progress_bar.progress(40)
                        audio_path, duration = extract_audio_from_video(video_path)

                        # 3. Speech Recognition
                        status_text.text("3/4: Transcribing speech and extracting timestamps (Whisper AI)...")
                        progress_bar.progress(70)
                        stt_result = transcribe_audio(audio_path, model_size=DEFAULT_WHISPER_MODEL)
                        raw_segments = stt_result["segments"]
                        detected_lang = stt_result["detected_language"]

                        # Cleanup temp audio and uploaded video
                        cleanup_temp_file(audio_path)
                        cleanup_temp_file(video_path)
                    else:
                        detected_lang = "en"
                        duration = sum(s["duration_seconds"] for s in raw_segments)

                    # 4. LSTM Sentiment Analysis
                    status_text.text("4/4: Running LSTM Deep Neural Network on speech segments...")
                    progress_bar.progress(90)
                    
                    analysis_result = analyzer.analyze_segments(raw_segments)
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()

                    # Save to session_state so results persist across downloads and reruns
                    st.session_state["video_analysis_result"] = analysis_result
                    st.session_state["video_analysis_duration"] = duration
                    st.session_state["video_analysis_lang"] = detected_lang
                    st.session_state["video_file_name"] = uploaded_file.name if uploaded_file else st.session_state.get("selected_scenario", "Demo Review")

                except Exception as e:
                    st.error(f"⚠️ Video Analysis Failed: {str(e)}")

        # ==========================================
        # DISPLAY RESULTS DASHBOARD (PERSISTENT)
        # ==========================================
        if st.session_state.get("video_analysis_result") is not None:
            analysis_result = st.session_state["video_analysis_result"]
            duration = st.session_state.get("video_analysis_duration", 0.0)

            st.success("✅ Video Speech Sentiment Analysis Complete!")
            st.markdown("---")

            overall_sentiment = analysis_result["overall_sentiment"]
            overall_conf = analysis_result["overall_confidence"]
            overall_probs = analysis_result["overall_probabilities"]
            analyzed_segs = analysis_result["analyzed_segments"]
            highlights = analysis_result["highlights"]
            timeline = analysis_result["timeline"]

            # 1. Hero Summary Banner
            st.markdown("### 🏆 Overall Video Sentiment")
            col_hero1, col_hero2 = st.columns([1, 1])

            with col_hero1:
                if overall_sentiment == "Positive":
                    st.markdown('<div class="sentiment-positive">🟢 OVERALL: POSITIVE</div>', unsafe_allow_html=True)
                elif overall_sentiment == "Neutral":
                    st.markdown('<div class="sentiment-neutral">🟡 OVERALL: NEUTRAL</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sentiment-negative">🔴 OVERALL: NEGATIVE</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Duration-Weighted Confidence:** `{overall_conf * 100:.1f}%`")
                st.progress(float(overall_conf))
                st.caption(f"Based on **{len(analyzed_segs)}** timestamped speech segments across **{duration:.1f}s**.")

            with col_hero2:
                st.markdown("#### Weighted Sentiment Probability Distribution")
                c_p1, c_p2, c_p3 = st.columns(3)
                c_p1.metric("🔴 Negative", f"{overall_probs['negative']*100:.1f}%")
                c_p2.metric("🟡 Neutral", f"{overall_probs['neutral']*100:.1f}%")
                c_p3.metric("🟢 Positive", f"{overall_probs['positive']*100:.1f}%")

                # Donut Chart for overall sentiment
                fig_donut, ax_donut = plt.subplots(figsize=(4.5, 2.2))
                labels = ["Negative", "Neutral", "Positive"]
                sizes = [overall_probs["negative"], overall_probs["neutral"], overall_probs["positive"]]
                colors = ["#ef4444", "#f59e0b", "#10b981"]
                wedges, texts, autotexts = ax_donut.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=140,
                    colors=colors,
                    wedgeprops=dict(width=0.45, edgecolor='w')
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
                plt.tight_layout()
                st.pyplot(fig_donut)
                plt.close()

            st.markdown("---")

            # 2. Key Highlights Section
            st.markdown("### 🌟 Key Video Highlights")
            col_h1, col_h2, col_h3 = st.columns(3)

            with col_h1:
                most_pos = highlights.get("most_positive", {})
                st.markdown(f"""
                <div class="highlight-card" style="border-left-color: #10b981;">
                    <b style="color: #065f46;">🌟 Most Positive Segment</b><br>
                    <small><b>⏱️ {most_pos.get('timestamp_label', '')}</b></small><br>
                    <i>"{most_pos.get('transcript', '')}"</i><br>
                    <b style="color: #10b981;">Positive Score: {most_pos.get('positive_probability', 0)*100:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

            with col_h2:
                most_neg = highlights.get("most_negative", {})
                st.markdown(f"""
                <div class="highlight-card" style="border-left-color: #ef4444;">
                    <b style="color: #991b1b;">⚠️ Most Negative Segment</b><br>
                    <small><b>⏱️ {most_neg.get('timestamp_label', '')}</b></small><br>
                    <i>"{most_neg.get('transcript', '')}"</i><br>
                    <b style="color: #ef4444;">Negative Score: {most_neg.get('negative_probability', 0)*100:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

            with col_h3:
                most_unc = highlights.get("most_uncertain", {})
                st.markdown(f"""
                <div class="highlight-card" style="border-left-color: #f59e0b;">
                    <b style="color: #92400e;">❓ Most Uncertain Segment</b><br>
                    <small><b>⏱️ {most_unc.get('timestamp_label', '')}</b></small><br>
                    <i>"{most_unc.get('transcript', '')}"</i><br>
                    <b style="color: #f59e0b;">Max Confidence: {most_unc.get('confidence', 0)*100:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 3. Interactive Sentiment Timeline
            st.markdown("### 📈 Interactive Sentiment Timeline")
            st.write("Tracks spoken sentiment transitions throughout the video duration:")

            if timeline:
                time_x = [pt["timestamp_seconds"] for pt in timeline]
                scores_y = [pt["sentiment_score"] for pt in timeline]
                hover_texts = [
                    f"<b>Time:</b> {pt['timestamp_label']}<br><b>Speech:</b> {pt['transcript_snippet']}<br><b>Sentiment:</b> {pt['sentiment']}<br><b>Confidence:</b> {pt['confidence']*100:.1f}%"
                    for pt in timeline
                ]
                
                fig_timeline = go.Figure()
                
                # Add zero line
                fig_timeline.add_shape(
                    type="line", line=dict(dash="dash", color="#cbd5e1", width=1.5),
                    x0=0, x1=max(time_x) + 2, y0=0, y1=0
                )

                # Add sentiment trace with golden line and sentiment markers
                fig_timeline.add_trace(go.Scatter(
                    x=time_x,
                    y=scores_y,
                    mode="lines+markers",
                    line=dict(color="#c59b27", width=3, shape="spline"),
                    marker=dict(size=10, color=["#10b981" if s>0.1 else ("#ef4444" if s<-0.1 else "#d4af37") for s in scores_y], line=dict(color="#ffffff", width=1.5)),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    name="Sentiment Trajectory"
                ))

                fig_timeline.update_layout(
                    xaxis_title="Video Timestamp (Seconds)",
                    yaxis_title="Sentiment Polarity (-1 Negative to +1 Positive)",
                    yaxis=dict(range=[-1.1, 1.1], tickvals=[-1, 0, 1], ticktext=["Negative (-1)", "Neutral (0)", "Positive (+1)"]),
                    height=350,
                    font=dict(family="Roboto, sans-serif", size=12),
                    margin=dict(l=40, r=40, t=20, b=40),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto, sans-serif")
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            st.markdown("---")

            # 4. Detailed Segment Breakdown Table
            st.markdown("### 📋 Segment-by-Segment Sentiment Analysis")
            
            seg_records = []
            for s in analyzed_segs:
                seg_records.append({
                    "Timestamp": s["timestamp_label"],
                    "Spoken Speech": s["transcript"],
                    "Sentiment": s["sentiment"],
                    "Confidence": f"{s['confidence']*100:.1f}%",
                    "Negative Prob": f"{s['negative_probability']*100:.1f}%",
                    "Neutral Prob": f"{s['neutral_probability']*100:.1f}%",
                    "Positive Prob": f"{s['positive_probability']*100:.1f}%"
                })
            
            df_segs = pd.DataFrame(seg_records)
            st.dataframe(df_segs, use_container_width=True)

            # 5. Download Exports
            st.markdown("### 📥 Download Reports")
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                # CSV Export
                export_df = pd.DataFrame(analyzed_segs)
                csv_buf = io.StringIO()
                export_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download Detailed CSV Report",
                    data=csv_buf.getvalue(),
                    file_name="sentiment_analysis.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            with col_d2:
                # Transcript TXT Export
                full_txt = "\n\n".join([f"[{s['timestamp_label']}] ({s['sentiment']} - {s['confidence']*100:.1f}%)\n{s['transcript']}" for s in analyzed_segs])
                st.download_button(
                    label="📄 Download Transcript TXT",
                    data=full_txt,
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # ==========================================
    # TAB 2: Input Testing & Explainability Studio
    # ==========================================
    with tab_test_suite:
        st.subheader("🔬 Interactive Input Testing & Explainability Studio")
        st.write("Perform deep-dive testing on text inputs to inspect token attribution, aspect extraction, model comparisons, and batch processing.")

        test_mode = st.radio(
            "Select Testing Mode:",
            ["🧪 Single Sentence Diagnosis & Explainability", "📂 Batch Testing & CSV Stress-Testing"],
            horizontal=True
        )

        if test_mode == "🧪 Single Sentence Diagnosis & Explainability":
            st.markdown("#### 1. Choose a Linguistic Challenge Preset or Enter Custom Text:")
            
            # Preset selector buttons
            cols_p1 = st.columns(3)
            cols_p2 = st.columns(3)
            
            preset_keys = list(TEST_PRESETS.keys())
            
            if "test_sentence_box" not in st.session_state:
                st.session_state["test_sentence_box"] = "The battery life is amazing and customer support was very helpful, but the screen scratches easily."

            for i, p_key in enumerate(preset_keys):
                target_col = cols_p1[i] if i < 3 else cols_p2[i - 3]
                if target_col.button(p_key, use_container_width=True):
                    st.session_state["test_sentence_box"] = TEST_PRESETS[p_key]
                    st.session_state["input_text_val"] = TEST_PRESETS[p_key]

            # Text input area
            user_input = st.text_area(
                "Input Sentence to Test:",
                height=90,
                key="test_sentence_box"
            )

            col_sub1, _ = st.columns([1, 2])
            with col_sub1:
                run_test_clicked = st.button("🔍 Run Full Sentiment Diagnosis & Explainability", type="primary", use_container_width=True)

            if run_test_clicked or user_input:
                if user_input.strip():
                    with st.spinner("Analyzing input representation, token contributions, and aspect decomposition..."):
                        diag = explain_prediction(user_input)

                    st.markdown("---")
                    
                    # 1. Hero Outcome Summary
                    sent = diag["sentiment"]
                    conf = diag["confidence"]
                    probs = diag["probabilities"]

                    st.markdown("### 🎯 Model Prediction & Probability Distribution")
                    col_res1, col_res2 = st.columns([1, 1.2])

                    with col_res1:
                        if sent == "Positive":
                            st.markdown('<div class="sentiment-positive">🟢 PREDICTED: POSITIVE</div>', unsafe_allow_html=True)
                        elif sent == "Neutral":
                            st.markdown('<div class="sentiment-neutral">🟡 PREDICTED: NEUTRAL</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="sentiment-negative">🔴 PREDICTED: NEGATIVE</div>', unsafe_allow_html=True)

                        st.markdown(f"**Confidence Level:** `{conf * 100:.2f}%`")
                        st.progress(float(conf))

                        st.markdown(f"""
                        - **Raw Input Length:** `{len(user_input)} characters`
                        - **Effective Words:** `{diag['pipeline_trace']['word_count']} words`
                        - **Padded Sequence Tokens:** `{diag['pipeline_trace']['sequence_length']} tokens`
                        """)

                    with col_res2:
                        # Softmax Probability Distribution Bar
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            x=["Negative", "Neutral", "Positive"],
                            y=[probs["negative"] * 100, probs["neutral"] * 100, probs["positive"] * 100],
                            marker_color=["#ef4444", "#f59e0b", "#10b981"],
                            text=[f"{probs['negative']*100:.1f}%", f"{probs['neutral']*100:.1f}%", f"{probs['positive']*100:.1f}%"],
                            textposition="auto"
                        ))
                        fig_bar.update_layout(
                            title="Softmax Probability Distribution (%)",
                            yaxis_title="Probability (%)",
                            yaxis=dict(range=[0, 105]),
                            height=240,
                            font=dict(family="JetBrains Mono, monospace", size=12),
                            margin=dict(l=20, r=20, t=35, b=20),
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("---")

                    # 2. Token-Level Attribution / Word Importance Highlighter
                    st.markdown("### 🧩 Word-Level Attribution & Sentiment Contribution")
                    st.write("Visualizes how individual words impact the neural network's final sentiment determination (green = positive pull, red = negative pull, gold/champagne = neutral/connector):")

                    tokens_attr = diag.get("token_attributions", [])
                    if tokens_attr:
                        chips_html = '<div style="background: #ffffff; padding: 1.2rem; border-radius: 12px; border: 1px solid #e8dfc8; box-shadow: 0 4px 14px rgba(212, 175, 55, 0.08); line-height: 2.2;">'
                        for t in tokens_attr:
                            w = t["word"]
                            pol = t["polarity"]
                            sc = t["impact_score"]
                            css_class = "word-pos" if pol == "positive" else ("word-neg" if pol == "negative" else "word-neu")
                            pol_icon = "🟢 +" if pol == "positive" else ("🔴 -" if pol == "negative" else "🟡")
                            chips_html += f'<span class="word-chip {css_class}" title="{t.get("explanation", "")} (Impact: {sc})">{w} <small style="opacity: 0.8;">{pol_icon}</small></span>'
                        chips_html += "</div>"
                        st.markdown(chips_html, unsafe_allow_html=True)

                        # Word attribution bar chart
                        st.markdown("<br><b>Word-by-Word Impact Breakdown:</b>", unsafe_allow_html=True)
                        word_df = pd.DataFrame(tokens_attr)
                        if not word_df.empty:
                            fig_attr = px.bar(
                                word_df,
                                x="word",
                                y="impact_score",
                                color="polarity",
                                color_discrete_map={"positive": "#10b981", "negative": "#ef4444", "neutral": "#d4af37"},
                                labels={"impact_score": "Sentiment Influence Score", "word": "Token", "polarity": "Polarity"},
                                height=260
                            )
                            fig_attr.update_layout(
                                font=dict(family="JetBrains Mono, monospace", size=12),
                                margin=dict(l=20, r=20, t=20, b=20),
                                plot_bgcolor="#ffffff",
                                paper_bgcolor="#ffffff"
                            )
                            st.plotly_chart(fig_attr, use_container_width=True)

                    # 3. Aspect-Based Sentiment Analysis Breakdown
                    st.markdown("---")
                    st.markdown("### 🏷️ Aspect-Based Sentiment Decomposition")
                    st.write("Detects domain-specific entities (e.g., Battery, Screen, Support, Pricing) and resolves sentiment per clause:")

                    aspects = diag.get("aspects", [])
                    if aspects:
                        for asp in aspects:
                            asp_name = asp["aspect"]
                            asp_sent = asp["sentiment"]
                            asp_class = "aspect-pos" if asp_sent == "Positive" else ("aspect-neg" if asp_sent == "Negative" else "aspect-neu")
                            st.markdown(f"""
                            <div style="background: #fdfcf7; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; border-left: 4px solid {'#10b981' if asp_sent=='Positive' else ('#ef4444' if asp_sent=='Negative' else '#d4af37')}; border-top: 1px solid #f2e9d2; border-right: 1px solid #f2e9d2; border-bottom: 1px solid #f2e9d2;">
                                <b>{asp_name}:</b> <span class="aspect-badge {asp_class}">{asp_sent}</span><br>
                                <small style="color: #786227;">Matched clause: <i>"{asp['clause']}"</i> {'(Negation detected)' if asp.get('negation_detected') else ''}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No specific entity aspect keywords detected in this sentence. Sentiment is evaluated globally.")

                    # 4. Model Comparison (LSTM vs Baseline TF-IDF + Logistic Regression)
                    st.markdown("---")
                    st.markdown("### ⚖️ Side-by-Side Model Comparison")
                    st.write("Compare the deep learning LSTM prediction with the traditional TF-IDF + Logistic Regression baseline on this exact input:")

                    base_comp = diag.get("baseline_comparison")
                    col_cmp1, col_cmp2 = st.columns(2)

                    with col_cmp1:
                        st.markdown("""
                        <div style="background: #ffffff; border-radius: 12px; padding: 1.3rem; border: 1px solid #e8dfc8; box-shadow: 0 4px 14px rgba(212, 175, 55, 0.1);">
                            <h4 style="margin-top:0; color: #996515; font-family: 'JetBrains Mono', monospace;">🧠 LSTM Deep Neural Network</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(f"**Predicted Sentiment:** `{diag['sentiment']}`")
                        st.markdown(f"**Confidence:** `{diag['confidence']*100:.2f}%`")
                        st.markdown(f"**Sequence Modeling:** Captures word order, long-range dependencies, and complex negations.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_cmp2:
                        st.markdown("""
                        <div style="background: #ffffff; border-radius: 12px; padding: 1.3rem; border: 1px solid #e8dfc8; box-shadow: 0 4px 14px rgba(212, 175, 55, 0.1);">
                            <h4 style="margin-top:0; color: #b8860b; font-family: 'JetBrains Mono', monospace;">📊 TF-IDF + Logistic Regression</h4>
                        """, unsafe_allow_html=True)
                        if base_comp:
                            st.markdown(f"**Predicted Sentiment:** `{base_comp['sentiment']}`")
                            st.markdown(f"**Confidence:** `{base_comp['confidence']*100:.2f}%`")
                            st.markdown(f"**Bag-of-Words Limitation:** Ignores sequential context beyond fixed n-grams.")
                        else:
                            st.info("Baseline model not loaded or trained yet.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # 5. Preprocessing Traceability Inspector
                    with st.expander("🛠️ View Step-by-Step Preprocessing Trace (Token IDs & Vectors)"):
                        trace = diag["pipeline_trace"]
                        st.markdown(f"""
                        - **Raw Input String:** `{trace['raw_input']}`
                        - **Cleaned Normalized String:** `{trace['cleaned_text']}`
                        - **Extracted Word Tokens:** `{diag['tokens']}`
                        - **Tokenizer Word IDs (1 to 10,000):** `{trace['token_ids']}`
                        - **Padded Sequence Dimensions:** `1 x {trace['max_length_limit']}`
                        """)

        else:
            # ==========================================
            # SUB-TAB: Batch Testing & CSV Stress-Testing
            # ==========================================
            st.markdown("#### 2. Batch Test Multiple Reviews Simultaneously")
            st.write("Paste multiple reviews (one per line) or upload a CSV file to evaluate throughput and review distribution.")

            batch_mode = st.radio("Input Source:", ["✍️ Paste Multiple Lines", "📁 Upload CSV File"], horizontal=True)

            batch_texts = []
            if batch_mode == "✍️ Paste Multiple Lines":
                default_batch = (
                    "The battery life is amazing and fast charging works well.\n"
                    "Terrible customer service, refused my refund request.\n"
                    "The package measures 10 inches and contains a user manual.\n"
                    "Not bad at all for the price, quite pleased with it.\n"
                    "The screen cracked on the first day, horrible durability."
                )
                raw_paste = st.text_area("Enter reviews (one per line):", value=default_batch, height=140)
                batch_texts = [line.strip() for line in raw_paste.split("\n") if line.strip()]

            else:
                batch_file = st.file_uploader("Upload CSV containing a 'review' column:", type=["csv"])
                if batch_file is not None:
                    try:
                        b_df = pd.read_csv(batch_file)
                        matching_cols = [c for c in b_df.columns if c.strip().lower() == "review"]
                        if matching_cols:
                            batch_texts = b_df[matching_cols[0]].dropna().astype(str).tolist()
                            st.success(f"Loaded **{len(batch_texts)}** review rows from CSV.")
                        else:
                            st.error("Uploaded CSV must contain a 'review' column.")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")

            if st.button("🚀 Run Batch Sentiment Evaluation", type="primary"):
                if not batch_texts:
                    st.warning("Please provide at least one review to evaluate.")
                else:
                    with st.spinner(f"Classifying {len(batch_texts)} reviews with LSTM Neural Network..."):
                        analyzer = SentimentPredictor()
                        results = analyzer.predict_batch(batch_texts)

                    res_df = pd.DataFrame(results)

                    # Distribution Metrics
                    st.markdown("### 📊 Batch Evaluation Summary")
                    total_count = len(res_df)
                    pos_count = int((res_df["sentiment"] == "Positive").sum())
                    neu_count = int((res_df["sentiment"] == "Neutral").sum())
                    neg_count = int((res_df["sentiment"] == "Negative").sum())

                    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                    col_b1.metric("Total Tested", total_count)
                    col_b2.metric("🟢 Positive", f"{pos_count} ({pos_count/total_count*100:.1f}%)")
                    col_b3.metric("🟡 Neutral", f"{neu_count} ({neu_count/total_count*100:.1f}%)")
                    col_b4.metric("🔴 Negative", f"{neg_count} ({neg_count/total_count*100:.1f}%)")

                    # Detailed Table
                    st.markdown("### 📋 Prediction Results")
                    st.dataframe(res_df, use_container_width=True)

                    # Download CSV
                    csv_buf = io.StringIO()
                    res_df.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="📥 Download Batch Results CSV",
                        data=csv_buf.getvalue(),
                        file_name="batch_sentiment_results.csv",
                        mime="text/csv",
                        type="primary"
                    )

    # ==========================================
    # TAB 3: Model Performance & Benchmarks
    # ==========================================
    with tab_metrics:
        st.subheader("Model Evaluation, Benchmarks & Error Analysis")
        metrics_data = load_metrics_data()
        
        if metrics_data:
            comp_table = metrics_data.get("comparison_table", [])
            st.markdown("### 🏆 Head-to-Head Comparison (Untouched Test Split)")
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### 🎯 Test Set Confusion Matrix")
                if CONFUSION_MATRIX_PLOT.exists():
                    st.image(str(CONFUSION_MATRIX_PLOT), use_container_width=True)
            with col_m2:
                st.markdown("#### 📉 Training Loss & Accuracy Learning Curves")
                if TRAINING_HISTORY_PLOT.exists():
                    st.image(str(TRAINING_HISTORY_PLOT), use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔬 Controlled Hyperparameter Search Table")
            if EXPERIMENTS_CSV_PATH.exists():
                st.dataframe(pd.read_csv(EXPERIMENTS_CSV_PATH), use_container_width=True)

            # Error Analysis Section
            st.markdown("---")
            st.markdown("### 🔍 Qualitative Error Analysis on Test Split")
            st.write("Detailed audit of misclassifications categorizing linguistic edge cases:")

            error_samples = metrics_data.get("error_analysis", [])
            if error_samples:
                err_df = pd.DataFrame(error_samples)
                
                # Category Filter
                categories = ["All Categories"] + sorted(list(err_df["error_category"].unique()))
                selected_cat = st.selectbox("Filter Error Category:", categories)

                if selected_cat != "All Categories":
                    filtered_err_df = err_df[err_df["error_category"] == selected_cat]
                else:
                    filtered_err_df = err_df

                st.dataframe(filtered_err_df, use_container_width=True)
            else:
                st.info("No misclassifications recorded on test split or error analysis not generated.")

    # ==========================================
    # TAB 4: Architecture & Deep Learning Guide
    # ==========================================
    with tab_arch:
        st.subheader("Deep Learning NLP & Speech Processing Pipeline")
        st.markdown("""
        ### 🔄 Sentiment Analyzer Full Pipeline Architecture
        ```
        Uploaded Video File (.mp4, .mov, .avi, .mkv, .webm)
                     │
                     ▼
        [Video Validation & Format Checks]  (Max size: 100MB, Max duration: 10m)
                     │
                     ▼
        [Audio Track Extraction]           (16kHz Mono PCM WAV via MoviePy / FFmpeg)
                     │
                     ▼
        [Speech Recognition (Whisper)]     (Extracts Timestamped Speech Segments)
                     │
                     ▼
        [Text Preprocessing & Cleaner]     (Negation-Aware Regex Normalizer)
                     │
                     ▼
        [Vocabulary Tokenizer & Padding]   (Vocabulary: 10,000, Max Sequence Length: 150)
                     │
                     ▼
        [LSTM Deep Neural Network]         (Embedding 128 -> LSTM 128 -> Dropout 0.5 -> Dense 64 -> Softmax 3)
                     │
                     ▼
        [Weighted Sentiment Aggregation]   (Duration & Word-Count Weighted Overall Score)
                     │
                     ▼
        [Video Dashboard & Timeline]       (Interactive Trajectory, Highlights & CSV Export)
        ```
        """)


if __name__ == "__main__":
    main()

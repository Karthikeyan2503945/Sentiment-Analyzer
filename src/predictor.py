"""
Unified Video Sentiment Prediction Pipeline for SentimentVision.
Orchestrates: Video Upload -> Validation -> Audio Extraction -> Whisper STT -> LSTM Sentiment Classification.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_WHISPER_MODEL,
    SUPPORTED_FORMATS,
    MAX_VIDEO_SIZE_MB
)
from src.video_processor import validate_media_file, extract_audio_from_video, cleanup_temp_file
from src.speech_to_text import transcribe_audio
from src.sentiment_analyzer import VideoSentimentAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VideoSentimentPipeline:
    """
    End-to-End Orchestrator for Video Sentiment Analysis.
    """
    def __init__(self, whisper_model_size: str = DEFAULT_WHISPER_MODEL):
        self.whisper_model_size = whisper_model_size
        self.analyzer = VideoSentimentAnalyzer()

    def process_video(
        self,
        video_path: Union[str, Path],
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute full pipeline on an uploaded video file.
        
        Pipeline Stages:
        1. Media Validation
        2. Audio Track Extraction
        3. Speech-to-Text Transcription with Timestamps (Whisper)
        4. LSTM Deep Learning Sentiment Classification & Aggregation
        5. Temporary File Cleanup
        
        Args:
            video_path: Path to the input video or audio file.
            language: Optional language code (e.g. 'en', 'auto').
            progress_callback: Optional UI status callback fn(stage_name, progress_ratio).
            
        Returns:
            Structured dictionary containing complete analysis and metrics.
        """
        video_path = Path(video_path)
        
        def update_progress(msg: str, val: float):
            if progress_callback:
                progress_callback(msg, val)
            logger.info(f"[{int(val*100)}%] {msg}")

        # Stage 1: Validation
        update_progress("Validating media format and file size...", 0.10)
        is_valid, error_msg, media_type = validate_media_file(video_path)
        if not is_valid:
            raise ValueError(error_msg)

        extracted_audio_path = None
        try:
            # Stage 2: Audio Extraction
            update_progress(f"Extracting high-fidelity audio stream from {media_type}...", 0.30)
            extracted_audio_path, duration = extract_audio_from_video(video_path)

            # Stage 3: Speech Recognition (Whisper)
            update_progress("Transcribing spoken speech and extracting timestamps (Whisper AI)...", 0.60)
            transcription_result = transcribe_audio(
                extracted_audio_path,
                model_size=self.whisper_model_size,
                language=language
            )

            # Stage 4: LSTM Sentiment Classification
            update_progress("Classifying speech segments with LSTM Deep Neural Network...", 0.85)
            sentiment_result = self.analyzer.analyze_segments(transcription_result["segments"])

            # Compile Complete Final Report
            update_progress("Finalizing video analytics and sentiment timeline...", 1.00)
            
            final_report = {
                "file_name": video_path.name,
                "media_type": media_type,
                "video_duration_seconds": round(duration, 2),
                "detected_language": transcription_result["detected_language"],
                "full_transcript": transcription_result["full_transcript"],
                "total_segments": sentiment_result["total_segments_analyzed"],
                "overall_sentiment": sentiment_result["overall_sentiment"],
                "overall_confidence": sentiment_result["overall_confidence"],
                "overall_probabilities": sentiment_result["overall_probabilities"],
                "highlights": sentiment_result["highlights"],
                "segments": sentiment_result["analyzed_segments"],
                "timeline": sentiment_result["timeline"]
            }

            return final_report

        finally:
            # Stage 5: Cleanup temp audio
            if extracted_audio_path:
                cleanup_temp_file(extracted_audio_path)


def analyze_video(
    video_path: Union[str, Path],
    whisper_model_size: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Public convenience API to analyze an uploaded video file.
    """
    pipeline = VideoSentimentPipeline(whisper_model_size=whisper_model_size)
    return pipeline.process_video(video_path, language=language, progress_callback=progress_callback)

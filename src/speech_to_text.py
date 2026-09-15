"""
Speech-to-Text Transcription Engine for SentimentVision.
Powered by OpenAI Whisper with timestamped segment extraction and multi-language detection.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_WHISPER_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS string (e.g., 65.5 -> 01:05)."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_segment_interval(start: float, end: float) -> str:
    """Format start and end seconds into 'MM:SS – MM:SS'."""
    return f"{format_timestamp(start)} – {format_timestamp(end)}"


class SpeechTranscriber:
    """
    Speech recognition engine utilizing Whisper.
    Caches model in memory for fast repeated transcription.
    """
    _model_cache = {}

    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL):
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """Load and cache Whisper model."""
        import whisper
        if self.model_size not in self._model_cache:
            logger.info(f"Loading Whisper model '{self.model_size}' (CPU optimized)...")
            self._model_cache[self.model_size] = whisper.load_model(self.model_size)
            logger.info(f"Whisper '{self.model_size}' model loaded successfully.")
        self.model = self._model_cache[self.model_size]

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file into full text and timestamped segments.
        
        Args:
            audio_path: Path to 16kHz mono WAV audio file.
            language: Optional language code (e.g. 'en').
            
        Returns:
            Dictionary containing:
                - full_transcript: str
                - language: str
                - segments: List[Dict[str, Any]] with start, end, text, duration, word_count
        """
        import whisper
        audio_path = str(audio_path)
        
        logger.info(f"Starting Whisper transcription for: {audio_path} (Language: {language or 'auto-detect'})")
        
        options = {}
        if language and language.lower() != "auto":
            options["language"] = language.lower()
            
        # Run Whisper transcription
        result = self.model.transcribe(audio_path, **options)
        
        full_text = result.get("text", "").strip()
        detected_language = result.get("language", "en")
        raw_segments = result.get("segments", [])
        
        if not full_text or len(raw_segments) == 0:
            raise ValueError("No speech could be detected in the audio track. The video may contain only silence or music.")

        parsed_segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            duration = max(end - start, 0.5)
            words = text.split()
            
            parsed_segments.append({
                "segment_id": len(parsed_segments) + 1,
                "start_time": start,
                "end_time": end,
                "timestamp_label": format_segment_interval(start, end),
                "transcript": text,
                "word_count": len(words),
                "duration_seconds": round(duration, 2),
            })

        if len(parsed_segments) == 0:
            raise ValueError("No speech could be detected in the audio track. The video may contain only silence or music.")

        logger.info(f"Transcription complete: {len(parsed_segments)} segments extracted. Detected Language: '{detected_language}'")
        
        return {
            "full_transcript": full_text,
            "detected_language": detected_language,
            "total_segments": len(parsed_segments),
            "segments": parsed_segments
        }


def transcribe_audio(
    audio_path: Union[str, Path],
    model_size: str = DEFAULT_WHISPER_MODEL,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for audio transcription.
    """
    transcriber = SpeechTranscriber(model_size=model_size)
    return transcriber.transcribe(audio_path, language=language)

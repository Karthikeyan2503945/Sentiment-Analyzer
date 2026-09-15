"""
Video Processing, Validation, and Audio Extraction Module for SentimentVision.
Handles video format validation, size checks, audio track extraction, and temp file management.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_FORMATS,
    MAX_VIDEO_SIZE_MB,
    MAX_VIDEO_DURATION_SEC,
    AUDIO_SAMPLE_RATE,
    TEMP_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_media_file(file_path: Union[str, Path], file_size_bytes: Optional[int] = None) -> Tuple[bool, str, str]:
    """
    Validate uploaded media file format and size.
    
    Args:
        file_path: Path to the file.
        file_size_bytes: Optional file size in bytes.
        
    Returns:
        Tuple of (is_valid: bool, error_message: str, media_type: str ['video'|'audio'|'unknown'])
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix not in SUPPORTED_FORMATS:
        return (
            False,
            f"Unsupported file format '{suffix}'. Supported formats are: {', '.join(SUPPORTED_FORMATS)}",
            "unknown"
        )
    
    # Check size
    if file_size_bytes is None and path.exists():
        file_size_bytes = path.stat().st_size
        
    if file_size_bytes is not None:
        size_mb = file_size_bytes / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            return (
                False,
                f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_VIDEO_SIZE_MB} MB.",
                "unknown"
            )
        if size_mb == 0:
            return (
                False,
                "Uploaded file is empty (0 bytes).",
                "unknown"
            )

    media_type = "video" if suffix in SUPPORTED_VIDEO_FORMATS else "audio"
    return True, "", media_type


def extract_audio_from_video(
    video_path: Union[str, Path],
    output_audio_path: Optional[Union[str, Path]] = None
) -> Tuple[str, float]:
    """
    Extract the audio stream from a video file and export as standard 16kHz mono WAV.
    
    Args:
        video_path: Path to input video file.
        output_audio_path: Optional output path for the extracted audio.
        
    Returns:
        Tuple of (audio_path: str, duration_seconds: float)
        
    Raises:
        ValueError: If video has no audio track, is corrupted, or exceeds duration limit.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Generate temp audio output path if not specified
    if output_audio_path is None:
        output_audio_path = TEMP_DIR / f"extracted_audio_{uuid.uuid4().hex[:8]}.wav"
    else:
        output_audio_path = Path(output_audio_path)

    # Cross-version MoviePy import supporting MoviePy 2.x and 1.x
    try:
        from moviepy import VideoFileClip, AudioFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
        except ImportError as e:
            raise ImportError(f"MoviePy is not installed or available: {e}")

    # Check if input is already an audio file
    if video_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
        logger.info(f"Input is already an audio format ({video_path.suffix}). Converting to target WAV...")
        audio_clip = None
        try:
            audio_clip = AudioFileClip(str(video_path))
            duration = audio_clip.duration
            if duration is not None and duration > MAX_VIDEO_DURATION_SEC:
                raise ValueError(f"Audio duration ({duration:.1f}s) exceeds maximum allowed limit of {MAX_VIDEO_DURATION_SEC}s.")
            audio_clip.write_audiofile(
                str(output_audio_path),
                fps=AUDIO_SAMPLE_RATE,
                nbytes=2,
                codec="pcm_s16le",
                ffmpeg_params=["-ac", "1"],
                logger=None
            )
            return str(output_audio_path), float(duration or 0.0)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to process audio file: {str(e)}")
        finally:
            if audio_clip is not None:
                try:
                    audio_clip.close()
                except Exception:
                    pass

    # Extract audio from video using moviepy
    video_clip = None
    try:
        try:
            video_clip = VideoFileClip(str(video_path))
        except Exception as e:
            raise ValueError(f"Unable to read video file. The file may be corrupted or in an unreadable format. Error: {str(e)}")

        duration = video_clip.duration
        if duration is None or duration <= 0:
            raise ValueError("Video duration is zero or unreadable.")
            
        if duration > MAX_VIDEO_DURATION_SEC:
            raise ValueError(f"Video duration ({duration:.1f}s) exceeds maximum limit of {MAX_VIDEO_DURATION_SEC}s ({MAX_VIDEO_DURATION_SEC // 60} minutes).")

        if video_clip.audio is None:
            raise ValueError("No audio track was detected in this video file. SentimentVision requires spoken audio.")

        logger.info(f"Extracting audio from video: {video_path.name} (Duration: {duration:.2f}s)...")
        
        # Write audio track to 16kHz mono PCM WAV
        video_clip.audio.write_audiofile(
            str(output_audio_path),
            fps=AUDIO_SAMPLE_RATE,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        logger.info(f"Audio successfully extracted to: {output_audio_path}")
        return str(output_audio_path), float(duration)

    finally:
        if video_clip is not None:
            try:
                if video_clip.audio is not None:
                    video_clip.audio.close()
                video_clip.close()
            except Exception:
                pass


def cleanup_temp_file(file_path: Union[str, Path]):
    """Safely remove a temporary file from disk."""
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink(missing_ok=True)
            logger.info(f"Cleaned up temporary file: {path.name}")
    except Exception as e:
        logger.warning(f"Could not delete temporary file {file_path}: {e}")

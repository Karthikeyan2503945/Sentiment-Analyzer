"""
Unit Tests for Video Validation, Audio Extraction, and File Constraints in SentimentVision.
"""

import os
import io
import pytest
from pathlib import Path
from src.config import SUPPORTED_VIDEO_FORMATS, SUPPORTED_AUDIO_FORMATS, MAX_VIDEO_SIZE_MB
from src.video_processor import validate_media_file, cleanup_temp_file


def test_supported_video_formats_validation(tmp_path):
    """Verify supported video formats are recognized as valid videos."""
    for fmt in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        test_file = tmp_path / f"test_sample{fmt}"
        test_file.write_bytes(b"mock_video_bytes_content")
        
        is_valid, msg, media_type = validate_media_file(test_file)
        assert is_valid is True
        assert msg == ""
        assert media_type == "video"


def test_supported_audio_formats_validation(tmp_path):
    """Verify supported audio formats are recognized as valid audio."""
    for fmt in [".wav", ".mp3", ".m4a", ".aac"]:
        test_file = tmp_path / f"test_audio{fmt}"
        test_file.write_bytes(b"mock_audio_bytes_content")
        
        is_valid, msg, media_type = validate_media_file(test_file)
        assert is_valid is True
        assert msg == ""
        assert media_type == "audio"


def test_unsupported_format_rejected(tmp_path):
    """Verify unallowed formats (e.g. .exe, .txt, .pdf) are rejected."""
    bad_file = tmp_path / "document.pdf"
    bad_file.write_bytes(b"dummy pdf content")
    
    is_valid, msg, media_type = validate_media_file(bad_file)
    assert is_valid is False
    assert "Unsupported file format" in msg
    assert media_type == "unknown"


def test_empty_file_rejected(tmp_path):
    """Verify 0-byte files are rejected with a clear message."""
    empty_file = tmp_path / "empty_video.mp4"
    empty_file.write_bytes(b"")
    
    is_valid, msg, media_type = validate_media_file(empty_file)
    assert is_valid is False
    assert "empty (0 bytes)" in msg


def test_oversized_file_rejected(tmp_path):
    """Verify files larger than MAX_VIDEO_SIZE_MB are rejected."""
    mock_file = tmp_path / "huge_video.mp4"
    oversized_bytes = (MAX_VIDEO_SIZE_MB + 5) * 1024 * 1024
    
    is_valid, msg, media_type = validate_media_file(mock_file, file_size_bytes=oversized_bytes)
    assert is_valid is False
    assert "exceeds maximum allowed limit" in msg


def test_cleanup_temp_file(tmp_path):
    """Verify temp file cleanup deletes files without raising exceptions."""
    temp_file = tmp_path / "temp_to_delete.wav"
    temp_file.write_bytes(b"temporary wav bytes")
    assert temp_file.exists()
    
    cleanup_temp_file(temp_file)
    assert not temp_file.exists()
    
    # Deleting non-existent file should be safe
    cleanup_temp_file(temp_file)


def test_extract_audio_from_audio_input(tmp_path):
    """Verify extract_audio_from_video handles audio files using MoviePy."""
    import numpy as np
    from moviepy.audio.AudioClip import AudioArrayClip
    from src.video_processor import extract_audio_from_video
    
    # Generate a brief 1.0 second silent mono audio track
    sample_rate = 16000
    dummy_samples = np.zeros((sample_rate, 1), dtype=np.float32)
    input_wav = tmp_path / "input_audio.wav"
    
    clip = AudioArrayClip(dummy_samples, fps=sample_rate)
    clip.write_audiofile(str(input_wav), fps=sample_rate, nbytes=2, codec="pcm_s16le", ffmpeg_params=["-ac", "1"], logger=None)
    clip.close()
    
    output_wav = tmp_path / "extracted.wav"
    extracted_path, duration = extract_audio_from_video(input_wav, output_audio_path=output_wav)
    
    assert Path(extracted_path).exists()
    assert duration > 0
    cleanup_temp_file(extracted_path)


def test_extract_audio_file_not_found():
    """Verify extract_audio_from_video raises FileNotFoundError for missing paths."""
    from src.video_processor import extract_audio_from_video
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video("non_existent_video_path.mp4")

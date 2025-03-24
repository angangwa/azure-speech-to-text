"""
Utility functions for Azure Speech Recognition application.
Contains helper functions used across different modules.
"""
import time
import logging
import soundfile as sf
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def get_audio_length(file_path: str) -> Optional[float]:
    """
    Get the length of an audio file in seconds

    Args:
        file_path (str): Path to the audio file

    Returns:
        Optional[float]: Length of audio file in seconds or None if error occurs
    """
    try:
        with sf.SoundFile(file_path) as f:
            length_seconds = f.frames / f.samplerate
            return length_seconds
    except Exception as e:
        logger.error(f"Error getting audio length: {e}")
        return None


def format_processing_info(
    audio_length: Optional[float],
    processing_time: Optional[float] = None,
    status_prefix: str = "Status:",
) -> str:
    """
    Format processing information string with audio length and processing time

    Args:
        audio_length (Optional[float]): Length of audio in seconds
        processing_time (Optional[float]): Processing time in seconds
        status_prefix (str): Prefix for the status message

    Returns:
        str: Formatted status message
    """
    length_info = ""
    time_info = ""

    if audio_length is not None:
        length_info = f"Audio length: {audio_length:.2f}s"

    if processing_time is not None:
        time_info = f"Processing time: {processing_time:.2f}s"

    parts = [part for part in [status_prefix, length_info, time_info] if part]
    return " | ".join(parts)

"""
OpenAI Whisper service for Azure Speech Recognition.
Implements calls to Azure OpenAI Whisper model for transcription.
"""
import time
import logging
from typing import Tuple, Optional

from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_ID,
)
from utils import get_audio_length, format_processing_info

logger = logging.getLogger(__name__)


def process_whisper_transcription(file_path: str) -> Tuple[str, str]:
    """
    Process audio file using Azure OpenAI Whisper model

    Args:
        file_path (str): Path to the audio file

    Returns:
        Tuple[str, str]: Status message and transcription result
    """
    if not file_path:
        return "Status: ❌ No file uploaded", ""

    try:
        # Get audio file length
        audio_length = get_audio_length(file_path)

        # Record start time
        start_time = time.time()

        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        logger.debug(f"Sending request to Azure OpenAI Whisper API")
        result = client.audio.transcriptions.create(
            file=open(file_path, "rb"), model=AZURE_OPENAI_DEPLOYMENT_ID
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        logger.info("Whisper transcription completed successfully")
        status = format_processing_info(
            audio_length, processing_time, "Status: ✅ Whisper Transcription complete"
        )
        return status, result.text

    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}")
        return "Status: ❌ Whisper Transcription error", str(e)

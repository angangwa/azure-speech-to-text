"""
Fast Transcription service for Azure Speech Recognition.
Implements direct API calls to Azure's Fast Transcription API.
"""
import time
import logging
import requests
import json
from typing import Tuple, Optional

from config import SPEECH_KEY, SERVICE_REGION
from utils import get_audio_length, format_processing_info

logger = logging.getLogger(__name__)


def process_fast_transcription(
    file_path: str, enable_diarization: bool = False, max_speakers: int = 2
) -> Tuple[str, str]:
    """
    Process audio file using Azure's Fast Transcription API

    Args:
        file_path (str): Path to the audio file
        enable_diarization (bool): Whether to enable speaker diarization
        max_speakers (int): Maximum number of speakers to identify

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

        url = f"https://{SERVICE_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
        headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY}

        # Prepare definition JSON with optional diarization
        definition = {"locales": ["en-US"]}

        # Add diarization settings if enabled
        if enable_diarization:
            definition["diarization"] = {"enabled": True, "maxSpeakers": max_speakers}

        # Create files for API request
        files = {
            "audio": open(file_path, "rb"),
            "definition": (None, json.dumps(definition)),
        }

        logger.debug(
            f"Sending request to Fast Transcription API with diarization={enable_diarization}, max_speakers={max_speakers}"
        )
        response = requests.post(url, headers=headers, files=files)

        # Calculate processing time
        processing_time = time.time() - start_time

        if response.status_code == 200:
            transcription_result = ""
            response_data = response.json()

            # Format results differently based on whether diarization is enabled
            if enable_diarization and "phrases" in response_data:
                last_speaker = None
                for phrase in response_data["phrases"]:
                    speaker = phrase.get("speaker", "Unknown")
                    if speaker != last_speaker:
                        transcription_result += f"\nSpeaker {speaker}: "
                        last_speaker = speaker
                    transcription_result += f"{phrase['text']} "
            else:
                for phrase in response_data.get("phrases", []):
                    transcription_result += phrase["text"] + "\n"

            logger.info("Fast transcription completed successfully")
            status = format_processing_info(
                audio_length, processing_time, "Status: ✅ Fast Transcription complete"
            )
            return status, transcription_result.strip()
        else:
            logger.error(
                f"Fast transcription failed with status code: {response.status_code}"
            )
            error_detail = f"API returned: {response.text}"
            status = format_processing_info(
                audio_length,
                processing_time,
                f"Status: ❌ Fast Transcription failed ({response.status_code})",
            )
            return status, error_detail

    except Exception as e:
        logger.error(f"Error during fast transcription: {e}")
        return "Status: ❌ Fast Transcription error", str(e)

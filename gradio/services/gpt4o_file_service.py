"""
GPT-4o File Transcription service for Azure Speech Recognition.
Implements calls to Azure OpenAI GPT-4o-transcribe model for file-based transcription.
"""
import time
import logging
from typing import Tuple, Optional, Dict, Any
import os

from openai import AzureOpenAI
from config import set_logging_level
from utils import get_audio_length, format_processing_info, process_transcription_with_confidence

logger = logging.getLogger(__name__)

# GPT-4o-transcribe configuration
AZURE_OPENAI_GPT4O_API_KEY = os.getenv("AZURE_OPENAI_GPT4O_API_KEY")
AZURE_OPENAI_GPT4O_ENDPOINT = os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT")
AZURE_OPENAI_GPT4O_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID")


def process_gpt4o_file_transcription(
    file_path: str, 
    prompt: str = "", 
    response_format: str = "text",
    include_logprobs: bool = False,
    visualization_format: str = "html"
) -> Tuple[str, str]:
    """
    Process audio file using Azure OpenAI GPT-4o-transcribe model

    Args:
        file_path (str): Path to the audio file
        prompt (str): Optional prompt to guide transcription
        response_format (str): Format of the response (text or json)
        include_logprobs (bool): Whether to include confidence scores
        visualization_format (str): How to visualize confidence scores (html, markdown, text)

    Returns:
        Tuple[str, str]: Status message and transcription result
    """
    if not file_path:
        return "Status: ❌ No file uploaded", ""
    
    if not all([AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, AZURE_OPENAI_GPT4O_DEPLOYMENT_ID]):
        return "Status: ❌ Missing GPT-4o API configuration", "Please set AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, and AZURE_OPENAI_GPT4O_DEPLOYMENT_ID in your .env file."

    try:
        # Get audio file length
        audio_length = get_audio_length(file_path)

        # Record start time
        start_time = time.time()

        # Create client for Azure OpenAI GPT-4o
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_GPT4O_API_KEY,
            api_version="2025-03-01-preview",  # Make sure to use the correct API version
            azure_endpoint=f"https://{AZURE_OPENAI_GPT4O_ENDPOINT.split('/openai/deployments')[0]}"
        )

        # Prepare parameters for the API call
        params = {
            "model": AZURE_OPENAI_GPT4O_DEPLOYMENT_ID,
            "file": open(file_path, "rb"),
            "response_format": "json" if include_logprobs else response_format
        }
        
        # Add optional parameters if provided
        if prompt:
            params["prompt"] = prompt
            
        # Add logprobs if requested
        if include_logprobs:
            params["include"] = ["logprobs"]

        logger.debug("Sending request to Azure OpenAI GPT-4o-transcribe API")
        result = client.audio.transcriptions.create(**params)

        # Calculate processing time
        processing_time = time.time() - start_time

        logger.info("GPT-4o transcription completed successfully")
        status = format_processing_info(
            audio_length, processing_time, "Status: ✅ GPT-4o Transcription complete"
        )
        
        # Process the result based on whether we requested confidence scores
        if include_logprobs:
            # Process the result with our confidence visualization
            formatted_result = process_transcription_with_confidence(
                result, format_type=visualization_format
            )
            return status, formatted_result
        else:
            # Return the text or full result based on format
            if response_format == "json":
                return status, str(result)
            else:
                return status, result

    except Exception as e:
        logger.error(f"Error during GPT-4o transcription: {e}")
        return "Status: ❌ GPT-4o Transcription error", str(e)
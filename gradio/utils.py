"""
Utility functions for the Azure Speech Recognition application.
"""
import time
import os
import soundfile as sf
import logging
import json
import numpy as np
import html
from typing import Dict, List, Any, Optional, Union, Tuple

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
    audio_length: Optional[float], processing_time: float, status_prefix: str
) -> str:
    """
    Format processing information for display

    Args:
        audio_length (float): Length of the audio file in seconds
        processing_time (float): Time taken to process the audio in seconds
        status_prefix (str): Prefix for the status message

    Returns:
        str: Formatted status message
    """
    if audio_length:
        length_info = f"Audio length: {audio_length:.2f}s"
        processing_speed = audio_length / processing_time if processing_time > 0 else 0
        speed_info = f"Processing speed: {processing_speed:.2f}x"
        time_info = f"Processing time: {processing_time:.2f}s"
        return f"{status_prefix} ({length_info}, {time_info}, {speed_info})"
    else:
        return f"{status_prefix} (Processing time: {processing_time:.2f}s)"


def process_transcription_with_confidence(
    transcription_result: Union[str, Dict, Any], 
    format_type: str = "html"
) -> str:
    """
    Process a transcription result with confidence scores and return formatted output

    Args:
        transcription_result: The transcription result from GPT-4o-transcribe
        format_type: The type of formatting to use ("html", "markdown", or "text")

    Returns:
        str: Formatted transcription with confidence scores
    """
    try:
        # If the result is already a string and not a JSON object
        if isinstance(transcription_result, str):
            try:
                # Try to parse it as JSON
                result = json.loads(transcription_result)
            except json.JSONDecodeError:
                # If it's not valid JSON, just return the string
                return transcription_result
        else:
            # If it's already a Python object
            result = transcription_result
            
        # Handle both direct OpenAI types and string representations
        if not isinstance(result, dict) and hasattr(result, 'text') and hasattr(result, 'logprobs'):
            text = result.text
            logprobs = result.logprobs
        elif isinstance(result, dict) and 'text' in result and 'logprobs' in result:
            text = result['text']
            logprobs = result['logprobs']
        else:
            # Can't find confidence scores, return plain text
            if hasattr(result, 'text'):
                return result.text
            elif isinstance(result, dict) and 'text' in result:
                return result['text']
            else:
                return str(result)
                
        # If we have no logprobs, just return the text
        if not logprobs:
            return text
            
        # Process based on format type
        if format_type == "html":
            return format_confidence_scores_html(text, logprobs)
        elif format_type == "markdown":
            return format_confidence_scores_markdown(text, logprobs)
        else:  # text
            return text
            
    except Exception as e:
        logger.error(f"Error processing transcription with confidence: {e}")
        # Return original transcription if there's an error
        if isinstance(transcription_result, str):
            return transcription_result
        elif hasattr(transcription_result, 'text'):
            return transcription_result.text
        else:
            return str(transcription_result)


def format_confidence_scores_html(text: str, logprobs: List) -> str:
    """
    Format transcription with HTML-based confidence visualization
    
    Args:
        text: The transcription text
        logprobs: List of log probability objects
        
    Returns:
        str: HTML formatted text with color-coded confidence scores
    """
    html_output = "<div class='confidence-text'>"
    
    # Create a map of positions to tokens and their confidence
    token_map = {}
    
    current_position = 0
    for lp in logprobs:
        # Get the token and its probability (convert from log probability)
        token = lp.token if hasattr(lp, 'token') else lp['token']
        log_prob = lp.logprob if hasattr(lp, 'logprob') else lp['logprob']
        
        # Convert log probability to probability (0-100%)
        probability = np.round(np.exp(log_prob) * 100, 2)
        
        # Store the token and its probability
        token_length = len(token)
        token_map[current_position] = {
            'token': token,
            'probability': probability,
            'length': token_length
        }
        current_position += token_length
    
    # Now create the HTML with colored spans
    processed_text = ""
    current_position = 0
    remaining_text = text
    
    for pos, token_info in sorted(token_map.items()):
        # Handle any text before this token
        if pos > current_position:
            prefix_text = remaining_text[:pos - current_position]
            processed_text += html.escape(prefix_text)
            remaining_text = remaining_text[pos - current_position:]
            current_position = pos
        
        # Process the token
        token = token_info['token']
        probability = token_info['probability']
        token_length = token_info['length']
        
        # Generate color based on probability (green for high confidence, red for low)
        r = int(255 * (1 - probability / 100))
        g = int(255 * (probability / 100))
        b = 0
        
        # Create a span with the color and a tooltip
        processed_text += f"<span style='color: rgb({r}, {g}, {b});' title='Confidence: {probability}%'>{html.escape(token)}</span>"
        
        # Update position and remaining text
        current_position += token_length
        remaining_text = remaining_text[token_length:]
    
    # Add any remaining text
    if remaining_text:
        processed_text += html.escape(remaining_text)
    
    html_output += processed_text
    html_output += "</div>"
    
    # Add CSS for better readability
    html_output += """
    <style>
    .confidence-text {
        font-family: sans-serif;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .confidence-text span {
        cursor: pointer;
    }
    </style>
    """
    
    return html_output


def format_confidence_scores_markdown(text: str, logprobs: List) -> str:
    """
    Format transcription for markdown displays (simpler version without colors)
    
    Args:
        text: The transcription text
        logprobs: List of log probability objects
        
    Returns:
        str: Markdown formatted text with confidence indicators
    """
    # Since we can't use color or tooltip in markdown, we'll simply return
    # the text with a summary of low confidence tokens at the end
    
    low_confidence_tokens = []
    
    for lp in logprobs:
        # Get the token and its probability
        token = lp.token if hasattr(lp, 'token') else lp['token']
        log_prob = lp.logprob if hasattr(lp, 'logprob') else lp['logprob']
        
        # Convert log probability to probability (0-100%)
        probability = np.round(np.exp(log_prob) * 100, 2)
        
        # Track low confidence tokens (less than 50% confidence)
        if probability < 50 and token.strip():
            low_confidence_tokens.append((token, probability))
    
    # Return the text with a note about low confidence tokens
    if low_confidence_tokens:
        result = text + "\n\n---\n**Low Confidence Words:**\n"
        for token, prob in low_confidence_tokens:
            result += f"- '{token.strip()}' ({prob}% confidence)\n"
        return result
    else:
        return text

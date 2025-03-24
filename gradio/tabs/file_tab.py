"""
File Input Tab for Azure Speech Recognition.
Implements the UI and functionality for speech recognition from audio files.
"""
import time
import gradio as gr
import logging
from typing import Tuple

from services.speech_recognition import speech_service
from utils import get_audio_length

logger = logging.getLogger(__name__)


def process_file(file_path, enable_diarization=False):
    """
    Process the uploaded audio file

    Args:
        file_path (str): Path to the audio file
        enable_diarization (bool): Whether to enable diarization

    Returns:
        Tuple[str, str, str]: Status, recognizing text, recognized text
    """
    if not file_path:
        return ("Status: ❌ No file uploaded", "", "")

    # Get audio file length
    audio_length = get_audio_length(file_path)
    speech_service.file_audio_length = audio_length
    length_info = (
        f"Audio length: {audio_length:.2f} seconds"
        if audio_length
        else "Could not determine audio length"
    )

    # Stop any ongoing recognition
    if speech_service.is_listening:
        speech_service.stop_microphone_recognition()
    if speech_service.is_file_processing:
        speech_service.stop_file_recognition()

    # Configure diarization settings
    speech_service.configure_diarization(enable=enable_diarization)

    success = speech_service.start_file_recognition(file_path)

    diarization_info = " with diarization" if enable_diarization else ""
    if success:
        return (
            f"Status: 📄 Processing file{diarization_info}... ({length_info})",
            "",
            "",
        )
    else:
        return ("Status: ❌ Failed to process file", "", "")


def refresh_file_ui():
    """
    Refresh the UI with the latest file recognition results and manage timer state

    Returns:
        Tuple: Status text, current recognizing, history, timer update
    """
    (
        status,
        current_recognizing,
        current_history,
    ) = speech_service.get_recognition_status()

    # Check if we should consider file processing as complete
    # We consider it complete if the session has been marked as stopped by the SDK
    if speech_service.is_file_processing and speech_service.file_session_stopped:
        logger.info("File processing detected as complete")
        speech_service.is_file_processing = False

    # Determine timer state based on processing status
    timer_update = gr.update(active=speech_service.is_file_processing)

    # Update status text based on processing state
    status_text = speech_service.get_file_processing_status()

    return status_text, current_recognizing, current_history, timer_update


def stop_file_processing():
    """
    Stop file processing and update UI

    Returns:
        Tuple: Updated UI components
    """
    speech_service.stop_file_recognition()
    status = "Status: ⏹️ File processing stopped"
    _, current_recognizing, current_history = speech_service.get_recognition_status()
    return status, current_recognizing, current_history, gr.update(active=False)


def display_file_info(file_path):
    """
    Display basic information about the uploaded file

    Args:
        file_path (str): Path to the audio file

    Returns:
        Tuple[str, str, str]: Status, recognizing text, history
    """
    if not file_path:
        return "Status: Ready to process file", "", ""

    # Get audio file length
    audio_length = get_audio_length(file_path)
    if audio_length:
        return (
            f"Status: File uploaded. Audio length: {audio_length:.2f} seconds",
            "",
            "",
        )
    else:
        return "Status: File uploaded. Could not determine audio length.", "", ""


def create_file_tab() -> gr.Tab:
    """
    Create the file input tab

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("File Input") as tab:
        # API information
        gr.Markdown(
            """
            **Continuous Speech Recognition using an audio file**
            """
        )

        # File processing section
        file_status_text = gr.Markdown("Status: Ready to process file")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Audio File", file_types=["audio"])

                # Add diarization options
                gr.Markdown("### Diarization Options")
                with gr.Column():
                    enable_diarization = gr.Checkbox(
                        label="Enable Speaker Diarization",
                        value=False,
                        info="Identify different speakers in the audio",
                    )

                process_button = gr.Button("Process File", variant="primary")
                file_clear_button = gr.Button("Clear Results")
                file_refresh_button = gr.Button("Refresh Results")
                stop_file_button = gr.Button("Stop Processing")

            with gr.Column(scale=2):
                file_recognizing_display = gr.Textbox(
                    label="Currently Recognizing",
                    placeholder="Waiting for file processing...",
                )
                file_recognized_display = gr.Textbox(
                    label="Recognition Results", lines=10
                )

        # Add timer for periodic updates
        file_timer = gr.Timer(value=0.1, active=False)

        # Show file information when a file is uploaded
        file_input.change(
            display_file_info,
            inputs=[file_input],
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
            ],
        )

        # Connect file processing functions with diarization parameters
        process_button.click(
            process_file,
            inputs=[file_input, enable_diarization],
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
            ],
        )

        # Start the timer when processing begins
        process_button.click(
            lambda: gr.update(active=True), inputs=None, outputs=[file_timer]
        )

        # Modified timer tick function to check if processing is complete and update UI
        file_timer.tick(
            refresh_file_ui,
            inputs=None,
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
                file_timer,
            ],
        )

        file_clear_button.click(
            lambda: ("Status: Results cleared", "", "", gr.update(active=False)),
            inputs=None,
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
                file_timer,
            ],
        )

        # Enhanced refresh button to update UI and timer status
        file_refresh_button.click(
            refresh_file_ui,
            inputs=None,
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
                file_timer,
            ],
        )

        # Enhanced stop button with consistent UI updates
        stop_file_button.click(
            stop_file_processing,
            inputs=None,
            outputs=[
                file_status_text,
                file_recognizing_display,
                file_recognized_display,
                file_timer,
            ],
        )

        # Instructions
        gr.Markdown(
            """
        ## Instructions
        1. Upload an audio file (WAV)
        2. Configure diarization options if needed:
           - Enable speaker diarization to identify different speakers
        3. Click 'Process File' to begin speech recognition
        4. The recognized text will appear in real-time with speaker identification (if enabled)
        5. If no text appears, try clicking 'Refresh Results'
        6. Click 'Stop Processing' to stop before completion
        7. Use 'Clear Results' to reset the recognition results
        """
        )

    return tab

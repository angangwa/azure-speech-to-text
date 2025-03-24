"""
Microphone Input Tab for Azure Speech Recognition.
Implements the UI and functionality for speech recognition from microphone.
"""
import gradio as gr
import logging
from typing import Tuple

from services.speech_recognition import speech_service

logger = logging.getLogger(__name__)


def toggle_recognition(enable_diarization=False):
    """
    Toggle between starting and stopping recognition

    Args:
        enable_diarization (bool): Whether to enable diarization
    """
    if not speech_service.is_listening:
        # Set diarization options before starting
        speech_service.configure_diarization(enable=enable_diarization)

        success = speech_service.start_microphone_recognition()
        if success:
            diarization_info = " with diarization" if enable_diarization else ""
            # Return active=True for the timer when starting listening
            return (
                gr.update(visible=True, value=f"Stop Listening{diarization_info}"),
                *speech_service.get_recognition_status(),
                gr.update(active=True),
            )
        else:
            return (
                gr.update(value="Start Listening"),
                "Status: ❌ Failed to start",
                "",
                "",
                gr.update(active=False),
            )
    else:
        success = speech_service.stop_microphone_recognition()
        # Show stopping state but keep timer active to continue refreshing UI
        return (
            gr.update(value="Stopping...", interactive=False),
            "Status: ⏳ Stopping recognition...",
            speech_service.recognizing_text,
            speech_service.recognized_history,
            gr.update(active=True),  # Keep timer active to refresh UI
        )


def refresh_ui():
    """Refresh the UI with the latest recognition results"""
    logger.debug("Refreshing UI")
    status, recognizing, history = speech_service.get_recognition_status()

    # Update button state and timer state based on service state
    if not speech_service.is_listening and not speech_service.is_stopping:
        # If not listening and not stopping, stop the timer and reset button
        button_update = gr.update(value="Start Listening", interactive=True)
        timer_update = gr.update(active=False)  # Stop the timer
    else:
        # If listening or stopping, keep timer active
        button_update = gr.update()
        timer_update = gr.update(active=True)

    # Return all expected values, including timer update
    return status, recognizing, history, button_update, timer_update


def clear_history():
    """Clear the recognition history"""
    speech_service.clear_history()
    return speech_service.get_recognition_status()


def create_microphone_tab() -> gr.Tab:
    """
    Create the microphone input tab

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("Microphone Input") as tab:
        # API information
        gr.Markdown(
            """
            **Continuous Speech Recognition using Microphone**
            """
        )

        # Status display
        status_text = gr.Markdown("Status: 🔇 Not listening")

        with gr.Row():
            with gr.Column(scale=1):
                # Add diarization options
                gr.Markdown("### Diarization Options")
                with gr.Column():
                    enable_diarization = gr.Checkbox(
                        label="Enable Speaker Diarization",
                        value=False,
                        info="Identify different speakers in the audio",
                    )

                listen_button = gr.Button("Start Listening", variant="primary")
                clear_button = gr.Button("Clear History")
                manual_refresh = gr.Button("Manual Refresh")

            with gr.Column(scale=2):
                # Display recognition results
                recognizing_display = gr.Textbox(
                    label="Currently Recognizing",
                    placeholder="Waiting for speech...",
                )
                recognized_display = gr.Textbox(label="Recognition History", lines=10)

        # Add timer for periodic updates (initially inactive)
        timer = gr.Timer(value=0.1, active=False)

        # Connect timer tick to refresh function - now also updates the timer itself
        timer.tick(
            refresh_ui,
            inputs=None,
            outputs=[
                status_text,
                recognizing_display,
                recognized_display,
                listen_button,
                timer,
            ],
        )

        # Button actions - updated to include diarization settings
        listen_button.click(
            toggle_recognition,
            inputs=[enable_diarization],
            outputs=[
                listen_button,
                status_text,
                recognizing_display,
                recognized_display,
                timer,
            ],
        )

        clear_button.click(
            clear_history,
            inputs=None,
            outputs=[status_text, recognizing_display, recognized_display],
        )

        # Update manual refresh to match the updated refresh_ui return values
        manual_refresh.click(
            refresh_ui,
            inputs=None,
            outputs=[
                status_text,
                recognizing_display,
                recognized_display,
                listen_button,
                timer,
            ],
        )

        # Instructions
        gr.Markdown(
            """
        ## Instructions
        1. Configure diarization options if needed:
           - Enable speaker diarization to identify different speakers
        2. Click 'Start Listening' to begin speech recognition
        3. Speak into your microphone
        4. Your speech will be recognized in real-time
        5. If no text appears, try clicking 'Manual Refresh'
        6. Click 'Stop Listening' when done
        7. Use 'Clear History' to reset the recognition history
        """
        )

    return tab

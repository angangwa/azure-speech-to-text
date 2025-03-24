"""
OpenAI Whisper Tab for Azure Speech Recognition.
Implements the UI and functionality for Azure OpenAI Whisper model.
"""
import gradio as gr
import logging

from services.whisper_service import process_whisper_transcription

logger = logging.getLogger(__name__)


def create_whisper_tab() -> gr.Tab:
    """
    Create the Whisper transcription tab

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("OpenAI Whisper") as tab:
        # Whisper transcription section
        whisper_status_text = gr.Markdown("Status: Ready for Whisper transcription")

        with gr.Row():
            with gr.Column(scale=1):
                whisper_file_input = gr.File(
                    label="Upload Audio File", file_types=["audio"]
                )
                whisper_process_button = gr.Button(
                    "Process with Whisper", variant="primary"
                )
                whisper_clear_button = gr.Button("Clear Results")

            with gr.Column(scale=2):
                whisper_transcription_display = gr.Textbox(
                    label="Transcription Results", lines=10
                )

        # Connect Whisper transcription functions
        whisper_process_button.click(
            process_whisper_transcription,
            inputs=[whisper_file_input],
            outputs=[whisper_status_text, whisper_transcription_display],
        )

        # Clear results
        whisper_clear_button.click(
            lambda: ("Status: Ready for Whisper transcription", ""),
            inputs=None,
            outputs=[whisper_status_text, whisper_transcription_display],
        )

        # Instructions
        gr.Markdown(
            """
            ## Whisper Transcription Instructions
            1. Upload an audio file (WAV)
            2. Click 'Process with Whisper' to send the file to Azure OpenAI Whisper model
            3. Once complete, the full transcript will appear in the results box
            4. This uses Azure OpenAI's latest Whisper model for high-quality transcription

            **Note**: This method may provide higher accuracy for complex audio content.
            """
        )

    return tab

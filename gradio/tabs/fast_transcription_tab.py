"""
Fast Transcription Tab for Azure Speech Recognition.
Implements the UI and functionality for Azure Fast Transcription API.
"""
import gradio as gr
import logging

from services.fast_transcription import process_fast_transcription

logger = logging.getLogger(__name__)


def create_fast_transcription_tab() -> gr.Tab:
    """
    Create the fast transcription tab

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("Fast Transcription") as tab:
        # Fast transcription section
        fast_status_text = gr.Markdown("Status: Ready for fast transcription")

        with gr.Row():
            with gr.Column(scale=1):
                fast_file_input = gr.File(
                    label="Upload Audio File", file_types=["audio"]
                )

                # Add diarization options using standard components
                gr.Markdown("### Diarization Options")
                with gr.Column():  # Use Column instead of Box
                    enable_diarization = gr.Checkbox(
                        label="Enable Speaker Diarization",
                        value=False,
                        info="Identify different speakers in the audio",
                    )
                    max_speakers = gr.Slider(
                        label="Maximum Number of Speakers",
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=2,
                        info="Set the maximum number of speakers to identify",
                    )

                fast_process_button = gr.Button(
                    "Process with Fast API", variant="primary"
                )
                fast_clear_button = gr.Button("Clear Results")

            with gr.Column(scale=2):
                fast_transcription_display = gr.Textbox(label="Transcription Results")

        # Connect fast transcription functions with diarization parameters
        fast_process_button.click(
            process_fast_transcription,
            inputs=[fast_file_input, enable_diarization, max_speakers],
            outputs=[fast_status_text, fast_transcription_display],
        )

        # Clear results
        fast_clear_button.click(
            lambda: ("Status: Ready for fast transcription", ""),
            inputs=None,
            outputs=[fast_status_text, fast_transcription_display],
        )

        # Instructions
        gr.Markdown(
            """
        ## Fast Transcription Instructions
        1. Upload an audio file (WAV)
        2. Configure diarization options:
           - Enable speaker diarization to identify different speakers in the audio
           - Set the maximum number of speakers you expect in the recording
        3. Click 'Process with Fast API' to send the file to Azure's Fast Transcription API
        4. Processing is typically much quicker than continuous recognition
        5. Once complete, the full transcript will appear in the results box

        **Note**: With diarization enabled, the output will identify which speaker said what.
        """
        )

    return tab

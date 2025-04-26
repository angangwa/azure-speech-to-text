"""
GPT-4o File Transcription Tab for Azure Speech Recognition.
Implements the UI and functionality for Azure OpenAI GPT-4o-transcribe file transcription.
"""
import gradio as gr
import logging

from services.gpt4o_file_service import process_gpt4o_file_transcription

logger = logging.getLogger(__name__)


def create_gpt4o_file_tab() -> gr.Tab:
    """
    Create the GPT-4o File Transcription tab

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("GPT-4o File Transcription") as tab:
        # GPT-4o file transcription section
        gpt4o_status_text = gr.Markdown("Status: Ready for GPT-4o transcription")

        with gr.Row():
            with gr.Column(scale=1):
                gpt4o_file_input = gr.File(
                    label="Upload Audio File", file_types=["audio"]
                )
                
                # Transcription options
                gr.Markdown("### Transcription Options")
                with gr.Column():
                    gpt4o_prompt = gr.Textbox(
                        label="Transcription Prompt (Optional)",
                        placeholder="Add context to guide transcription, e.g., 'This is a call center conversation about banking'"
                    )
                    
                    include_logprobs = gr.Checkbox(
                        label="Include Confidence Scores",
                        value=False,
                        info="Include token-level confidence scores"
                    )
                    
                    visualization_format = gr.Radio(
                        label="Confidence Score Visualization",
                        choices=["html", "markdown", "text"],
                        value="html",
                        info="HTML: Color-coded with tooltips, Markdown: Text with summary, Text: Plain text",
                        visible=False  # Initially hidden, will show when confidence scores are enabled
                    )

                gpt4o_process_button = gr.Button(
                    "Process with GPT-4o", variant="primary"
                )
                gpt4o_clear_button = gr.Button("Clear Results")

            with gr.Column(scale=2):
                gpt4o_transcription_display = gr.HTML(
                    label="Transcription Results",
                    value=""
                )

        # Show/hide visualization format options based on confidence scores checkbox
        include_logprobs.change(
            lambda x: gr.update(visible=x),
            inputs=[include_logprobs],
            outputs=[visualization_format]
        )

        # Connect GPT-4o transcription functions
        def process_gpt4o_with_options(file_path, prompt, include_probs, viz_format):
            if not file_path:
                return "Status: ❌ No file uploaded", ""
                
            status, result = process_gpt4o_file_transcription(
                file_path=file_path,
                prompt=prompt,
                response_format="text",
                include_logprobs=include_probs,
                visualization_format=viz_format
            )
            
            return status, result
                
        gpt4o_process_button.click(
            process_gpt4o_with_options,
            inputs=[gpt4o_file_input, gpt4o_prompt, include_logprobs, visualization_format],
            outputs=[gpt4o_status_text, gpt4o_transcription_display],
        )

        # Clear results
        gpt4o_clear_button.click(
            lambda: ("Status: Ready for GPT-4o transcription", ""),
            inputs=None,
            outputs=[gpt4o_status_text, gpt4o_transcription_display],
        )

        # Instructions
        gr.Markdown(
            """
            ## GPT-4o Transcription Instructions
            1. Upload an audio file (WAV, MP3, M4A, etc.)
            2. Configure transcription options:
               - Add an optional prompt to guide the transcription with context
               - Enable confidence scores to see color-coded word confidence (hover over words to see exact percentages)
               - If confidence scores are enabled, choose visualization format:
                 - **HTML**: Words are color-coded (green=high confidence, red=low) with tooltips
                 - **Markdown**: Plain text with summary of low-confidence words at the end
                 - **Text**: Plain text only
            3. Click 'Process with GPT-4o' to send the file to Azure OpenAI GPT-4o-transcribe model
            4. Once complete, the full transcript will appear in the results box

            **Note**: This uses Azure OpenAI's GPT-4o-transcribe model for high-quality transcription and is separate from the Whisper model.

            **Requirements**: This feature requires AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, and AZURE_OPENAI_GPT4O_DEPLOYMENT_ID in your .env file.
            """
        )

    return tab
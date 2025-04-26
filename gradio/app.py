"""
Main application for Azure Speech Recognition.
Combines all tabs and components and launches the Gradio interface.
"""
import gradio as gr
import logging
import argparse

from config import (
    verify_configs,
    SPEECH_KEY,
    SERVICE_REGION,
    set_logging_level,
    get_current_logging_level,
)
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_ID,
)

from tabs.microphone_tab import create_microphone_tab
from tabs.file_tab import create_file_tab
from tabs.fast_transcription_tab import create_fast_transcription_tab
from tabs.whisper_tab import create_whisper_tab
from tabs.gpt4o_file_tab import create_gpt4o_file_tab
from tabs.gpt4o_realtime_tab import create_gpt4o_realtime_tab

logger = logging.getLogger(__name__)


def toggle_debug_mode(enable_debug):
    """Toggle debug logging mode"""
    level = set_logging_level(debug_mode=enable_debug)
    return f"Current logging level: {logging.getLevelName(level)}"


def create_app():
    """
    Create and configure the Gradio app

    Returns:
        gr.Blocks: Configured Gradio application
    """
    # Create Gradio interface
    with gr.Blocks(title="Azure Speech Recognition") as demo:
        gr.Markdown("# Azure Speech Recognition")

        with gr.Tabs():
            # Add all tabs
            create_microphone_tab()
            create_file_tab()
            create_fast_transcription_tab()
            create_whisper_tab()
            create_gpt4o_file_tab()
            create_gpt4o_realtime_tab()

        # Debug section
        with gr.Accordion("Debug Information", open=False):
            configs = verify_configs()
            debug_info = gr.Markdown(
                f"""
            - Speech Key Available: {bool(SPEECH_KEY)}
            - Service Region: {SERVICE_REGION}
            - Azure OpenAI API Key Available: {bool(AZURE_OPENAI_API_KEY)}
            - Azure OpenAI Endpoint Available: {bool(AZURE_OPENAI_ENDPOINT)}
            - Azure OpenAI Deployment ID: {AZURE_OPENAI_DEPLOYMENT_ID}
            - Check console for detailed logs
            """
            )

            # Add debug mode toggle
            gr.Markdown("### Logging Control")
            debug_status = gr.Markdown(
                f"Current logging level: {get_current_logging_level()}"
            )
            debug_toggle = gr.Checkbox(
                label="Enable Debug Logging",
                value=get_current_logging_level() == "DEBUG",
                info="Toggle to enable/disable detailed debug logging",
            )
            debug_toggle.change(
                toggle_debug_mode, inputs=[debug_toggle], outputs=[debug_status]
            )

    return demo


def main():
    """Main entry point for the application"""
    try:
        # Add command line argument parsing for debug mode
        parser = argparse.ArgumentParser(description="Azure Speech Recognition App")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        args = parser.parse_args()

        # Set logging level based on command-line argument
        set_logging_level(debug_mode=args.debug)

        # Verify configurations
        configs = verify_configs()

        # Create app
        demo = create_app()

        # Check if credentials are available and launch app
        if not configs["Speech Service"]:
            logger.error("Missing SPEECH_KEY or SERVICE_REGION environment variables.")
            print(
                "Missing SPEECH_KEY or SERVICE_REGION environment variables. Please check your .env file."
            )
            return

        if not configs["Azure OpenAI"]:
            logger.warning(
                "Missing Azure OpenAI credentials. Whisper transcription will not be available."
            )
            print(
                "Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT environment variables. Whisper tab will not work."
            )

        logger.info("Starting Gradio app")
        demo.launch(share=False)

    except Exception as e:
        logger.error(f"Error starting application: {e}")
        print(f"Error starting application: {e}")


if __name__ == "__main__":
    main()

"""
GPT-4o Real-time Transcription Tab for Azure Speech Recognition.
Implements the UI and functionality for real-time transcription using WebSockets with GPT-4o-transcribe.
"""
import gradio as gr
import logging
import os
import sys
import time
import asyncio
from typing import Dict, Any

# Add the notebooks directory to path
notebooks_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../notebooks"))
if notebooks_dir not in sys.path:
    sys.path.append(notebooks_dir)

# Import the streaming service
from services.streaming_transcription_service import async_stream_transcription

logger = logging.getLogger(__name__)


def create_gpt4o_realtime_tab() -> gr.Tab:
    """
    Create the GPT-4o Real-time Transcription tab using WebSockets

    Returns:
        gr.Tab: Gradio tab component
    """
    with gr.Tab("GPT-4o Real-time Transcription") as tab:
        # Real-time transcription section
        realtime_status_text = gr.Markdown("Status: Ready for real-time transcription")

        with gr.Row():
            with gr.Column(scale=1):
                # Configuration options
                gr.Markdown("### Service Configuration")
                with gr.Column():
                    service_type = gr.Radio(
                        label="Service Type",
                        choices=["azure", "openai"],
                        value="azure",
                        info="Choose between Azure OpenAI or direct OpenAI API"
                    )
                    
                    model_choice = gr.Radio(
                        label="Model",
                        choices=["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
                        value="gpt-4o-transcribe",
                        info="Choose between standard or mini model"
                    )
                
                gr.Markdown("### Audio Processing Options")
                with gr.Column():
                    noise_reduction = gr.Radio(
                        label="Noise Reduction",
                        choices=["none", "near_field", "far_field"],
                        value="none",
                        info="'none': no reduction, 'near_field': for close mics, 'far_field': for distant mics"
                    )
                    
                    turn_threshold = gr.Slider(
                        label="Voice Activity Detection Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                        info="Higher values need louder speech to trigger (0.0-1.0)"
                    )
                    
                    max_duration = gr.Slider(
                        label="Recording Duration (seconds)",
                        minimum=10,
                        maximum=300,
                        step=10,
                        value=60,
                        info="The recording will automatically stop after this duration"
                    )

                # Add a message about recording time
                duration_info = gr.Markdown("Recording will run for **60 seconds** unless stopped manually")
                
                start_button = gr.Button("Start Recording", variant="primary")
                stop_button = gr.Button("Stop Recording", variant="stop")
                clear_button = gr.Button("Clear Results")

            with gr.Column(scale=2):
                # Display transcription results
                transcription_history = gr.Textbox(
                    label="Transcription", 
                    lines=15,
                    placeholder="Transcription will appear here while you speak"
                )

        # Global state to store transcription results and control
        state = gr.State({
            "is_recording": False,
            "current_text": "",
            "history": [],
            "termination_event": None,  # AsyncEvent for signaling termination
            "loop": None,  # Reference to the event loop
            "cancel_requested": False   # Flag for cancellation
        })
        
        # Function to clear the transcription history
        def clear_results(state_dict):
            """Clear all transcription results"""
            state_dict["current_text"] = ""
            state_dict["history"] = []
            return "Status: 🧹 Transcription history cleared", "", state_dict
        
        # Update the duration info when the slider changes
        def update_duration_info(duration):
            """Update the duration info message based on slider value"""
            return f"Recording will run for **{duration} seconds** unless stopped manually"
        
        # Connect the duration slider to the info update
        max_duration.change(
            fn=update_duration_info,
            inputs=[max_duration],
            outputs=[duration_info]
        )
        
        # Function to actually start the async transcription process
        async def run_async_transcription(
            service_type_val, model_val, noise_red_val, threshold_val, duration_val, state_dict
        ):
            """Run the async streaming transcription and update the UI"""
            try:
                # Create an asyncio event for termination signaling
                termination_event = asyncio.Event()
                state_dict["termination_event"] = termination_event
                state_dict["cancel_requested"] = False
                
                # Store the current event loop
                state_dict["loop"] = asyncio.get_event_loop()
                
                # Set recording flag
                state_dict["is_recording"] = True
                
                # Convert noise reduction setting
                if noise_red_val == "none":
                    noise_red_val = None
                
                logger.debug(f"Starting async transcription with service_type={service_type_val}, model={model_val}")
                
                # Process streaming events
                status_text = f"Status: 🎙️ Recording in progress... (will run for {duration_val} seconds)"
                history_text = ""
                
                # Set up a generator that will exit when termination is requested
                async def event_generator():
                    try:
                        async for event in async_stream_transcription(
                            service_type=service_type_val,
                            model=model_val,
                            noise_reduction=noise_red_val,
                            turn_threshold=threshold_val,
                            include_logprobs=True,  # Always include for proper operation
                            duration=duration_val
                        ):
                            # Check if termination was requested
                            if termination_event.is_set() or state_dict.get("cancel_requested", False):
                                logger.debug("Termination requested, breaking event loop")
                                break
                            
                            # Yield the event
                            yield event
                    except Exception as e:
                        logger.error(f"Error in event generator: {e}", exc_info=True)
                        # Yield an error event
                        yield {
                            "event_type": "error",
                            "data": f"Error in streaming: {str(e)}",
                            "timestamp": time.time()
                        }
                
                # Process events and update UI
                has_started = False
                async for event in event_generator():
                    event_type = event.get("event_type")
                    logger.debug(f"Received event: {event_type}")
                    
                    # Mark that we've started receiving events
                    has_started = True
                    
                    # Check for termination again
                    if termination_event.is_set() or state_dict.get("cancel_requested", False):
                        logger.debug("Termination check during event processing - breaking loop")
                        break
                    
                    if event_type == "delta":
                        # Incremental transcription update - aggregate in current_text
                        delta = event.get("data", "")
                        current_text = event.get("current_text", state_dict["current_text"] + delta)
                        state_dict["current_text"] = current_text
                        
                        # Log the delta if significant
                        if len(delta.strip()) > 0:
                            logger.debug(f"Delta: '{delta}', Current: '{current_text[:30]}...'")
                            
                    elif event_type == "transcript":
                        # Completed transcript segment
                        transcript = event.get("data", "")
                        logger.debug(f"Completed transcript: '{transcript[:30]}...'")
                        
                        # Add to history
                        if transcript.strip():
                            state_dict["history"].append(transcript)
                            history_text = "\n".join(state_dict["history"])
                            
                            # Yield the updates - only update on complete transcripts
                            yield status_text, history_text, state_dict
                            
                        # Reset current text
                        state_dict["current_text"] = ""
                        
                    elif event_type == "status":
                        # Status update
                        status_msg = event.get("data", "")
                        logger.debug(f"Status update: {status_msg}")
                        
                        # Update status if needed
                        if "speech detected" in status_msg.lower():
                            status_text = f"Status: 🗣️ Speech detected, listening... ({duration_val - (time.time() - event.get('timestamp', time.time())):.0f}s remaining)"
                            # Yield status update
                            yield status_text, history_text, state_dict
                        elif "speech stopped" in status_msg.lower():
                            status_text = f"Status: 🎙️ Waiting for speech... ({duration_val - (time.time() - event.get('timestamp', time.time())):.0f}s remaining)"
                            # Yield status update
                            yield status_text, history_text, state_dict
                        elif "error" in status_msg.lower() or "closed" in status_msg.lower():
                            status_text = f"Status: ⚠️ {status_msg}"
                            # Yield status update
                            yield status_text, history_text, state_dict
                        
                    elif event_type == "error":
                        # Error message
                        error_msg = event.get("data", "")
                        logger.error(f"Error in transcription: {error_msg}")
                        
                        # Update status
                        status_text = f"Status: ❌ Error: {error_msg}"
                        
                        # Yield the updates
                        yield status_text, history_text, state_dict
                
                # Check if we started but didn't receive any events
                if not has_started:
                    logger.warning("No events received from transcription service")
                    yield "Status: ⚠️ No events received from transcription service", history_text, state_dict
                
                # Done processing, update final state
                state_dict["is_recording"] = False
                if termination_event.is_set() or state_dict.get("cancel_requested", False):
                    status_text = "Status: ⏹️ Recording stopped by user"
                else:
                    status_text = "Status: ✅ Recording complete"
                
                # Yield final state
                yield status_text, history_text, state_dict
                
            except Exception as e:
                # Log and handle any exceptions
                logger.error(f"Error in transcription: {e}", exc_info=True)
                state_dict["is_recording"] = False
                status_text = f"Status: ❌ Error: {str(e)}"
                history_text = "\n".join(state_dict.get("history", []))
                
                # Yield error state
                yield status_text, history_text, state_dict
            finally:
                # Ensure state is cleaned up
                state_dict["is_recording"] = False
                state_dict["termination_event"] = None
                state_dict["loop"] = None
                state_dict["cancel_requested"] = False
        
        # Function called when the start button is clicked
        def start_recording(
            service_type_val, model_val, noise_red_val, threshold_val, duration_val, state_dict
        ):
            """Start recording and transcription"""
            # Check if already recording
            if state_dict.get("is_recording", False):
                return "Status: ⚠️ Already recording", \
                       "\n".join(state_dict.get("history", [])), state_dict
            
            # Reset state
            state_dict["current_text"] = ""
            state_dict["history"] = []
            state_dict["cancel_requested"] = False
            
            logger.debug(f"Starting recording with service_type={service_type_val}, model={model_val}")
            
            # Return initial status and state - this will trigger the async function
            return gr.update(value=f"Status: 🎙️ Starting recording... (will run for {duration_val} seconds)"), "", state_dict
        
        # Function called when the stop button is clicked
        def stop_recording(state_dict):
            """Stop the current recording"""
            # Check if recording
            if not state_dict.get("is_recording", False):
                return "Status: Not currently recording", \
                       "\n".join(state_dict.get("history", [])), state_dict
            
            # Mark cancellation requested immediately
            state_dict["cancel_requested"] = True
            
            # Signal termination if possible
            termination_event = state_dict.get("termination_event")
            loop = state_dict.get("loop")
            
            if termination_event and loop:
                try:
                    # Set the event - need to use call_soon_threadsafe from the loop
                    logger.debug("Signaling termination of recording via event")
                    loop.call_soon_threadsafe(termination_event.set)
                except Exception as e:
                    logger.error(f"Error setting termination event: {e}")
            
            # Update state - mark recording as stopped even if event setting failed
            state_dict["is_recording"] = False
            
            return "Status: ⏹️ Stopping recording...", \
                   "\n".join(state_dict.get("history", [])), state_dict
        
        # Connect the buttons to functions
        start_button.click(
            fn=start_recording,
            inputs=[
                service_type,
                model_choice,
                noise_reduction,
                turn_threshold,
                max_duration,
                state
            ],
            outputs=[
                realtime_status_text,
                transcription_history,
                state
            ]
        ).then(  # Chain the async function after the start function
            fn=run_async_transcription,
            inputs=[
                service_type,
                model_choice,
                noise_reduction,
                turn_threshold,
                max_duration,
                state
            ],
            outputs=[
                realtime_status_text,
                transcription_history,
                state
            ]
        )
        
        stop_button.click(
            fn=stop_recording,
            inputs=[state],
            outputs=[
                realtime_status_text,
                transcription_history,
                state
            ]
        )
        
        clear_button.click(
            fn=clear_results,
            inputs=[state],
            outputs=[
                realtime_status_text,
                transcription_history,
                state
            ]
        )

        # Instructions
        gr.Markdown(
            """
            ## GPT-4o Real-time Transcription Instructions
            1. Configure your transcription settings:
               - **Service Type**: Choose between Azure OpenAI or direct OpenAI service
               - **Model**: Select between standard or mini model based on needs
               - **Noise Reduction**: Choose an option based on your microphone setup
               - **Voice Activity Detection**: Adjust sensitivity for speech detection
               - **Recording Duration**: Set how long the recording session will run before automatically stopping
            
            2. Click 'Start Recording' to begin transcription
            3. Speak into your microphone
            4. Your speech will be transcribed in real-time with GPT-4o-transcribe
            5. The recording will automatically stop after the specified duration
            6. Click 'Stop Recording' to end the session early
            7. Use 'Clear Results' to reset the transcription history
            
            **Note**: This feature uses WebSockets to stream audio directly to the GPT-4o-transcribe model, providing fast and accurate real-time transcription.
            
            **Requirements**: 
            - For Azure: AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, and AZURE_OPENAI_GPT4O_DEPLOYMENT_ID
            - For direct OpenAI: OPENAI_API_KEY
            """
        )

    return tab
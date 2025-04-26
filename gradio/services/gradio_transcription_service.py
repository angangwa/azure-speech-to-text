"""
Enhanced Transcription Service for Gradio Integration.
This service extends the core TranscriptionService to work better with Gradio's UI update model.
"""
import os
import sys
import json
import asyncio
import threading
import logging
import time
import queue
from typing import Optional, List, Tuple, Dict, Any, Callable

# Add notebooks directory to path so we can import the transcription service
notebooks_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../notebooks"))
if notebooks_dir not in sys.path:
    sys.path.append(notebooks_dir)

# Import transcription service
from transcription_websocket_service import TranscriptionService

# Get configuration from environment variables directly
AZURE_OPENAI_GPT4O_API_KEY = os.getenv("AZURE_OPENAI_GPT4O_API_KEY")
AZURE_OPENAI_GPT4O_ENDPOINT = os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT")
AZURE_OPENAI_GPT4O_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

# Check if we have the required credentials and log warnings if not
if not all([AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, AZURE_OPENAI_GPT4O_DEPLOYMENT_ID]):
    logger.warning("Missing some Azure OpenAI GPT-4o credentials, the Azure transcription service might not work properly")

if not OPENAI_API_KEY:
    logger.warning("Missing OpenAI API key, the OpenAI transcription service might not work properly")


class GradioTranscriptionService(TranscriptionService):
    """
    Extended TranscriptionService with Gradio-specific enhancements.
    This class adds:
    1. Real-time UI update capabilities
    2. Better termination control
    3. State management for Gradio integration
    """
    
    def __init__(
        self,
        service_type: str = "azure",
        model: str = "gpt-4o-transcribe",
        noise_reduction: Optional[str] = None,
        turn_threshold: float = 0.5,
        turn_prefix_padding_ms: int = 300,
        turn_silence_duration_ms: int = 500,
        include_logprobs: bool = True,
        max_duration: int = 60,
        update_callback: Optional[Callable[[str, str, str], None]] = None,
        **kwargs
    ):
        """
        Initialize the Gradio-enhanced transcription service.
        
        Args:
            service_type: "azure" or "openai"
            model: Model name ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
            noise_reduction: Type of noise reduction (None, "near_field", or "far_field")
            turn_threshold: Voice activity detection threshold (0.0 to 1.0)
            turn_prefix_padding_ms: Padding time before speech detection (ms)
            turn_silence_duration_ms: Silent time to end a turn (ms)
            include_logprobs: Whether to include confidence scores
            max_duration: Maximum recording duration in seconds
            update_callback: Optional callback for UI updates
            **kwargs: Additional service parameters (endpoint, deployment, api_key)
        """
        # Initialize parent TranscriptionService
        super().__init__(
            service_type=service_type,
            model=model,
            noise_reduction=noise_reduction,
            turn_threshold=turn_threshold,
            turn_prefix_padding_ms=turn_prefix_padding_ms,
            turn_silence_duration_ms=turn_silence_duration_ms,
            include_logprobs=include_logprobs,
            **kwargs
        )
        
        # Gradio-specific state
        self.update_callback = update_callback
        self.max_duration = max_duration
        self.termination_event = threading.Event()
        self.main_thread = None
        self.websocket_task = None
        self.gradio_state = {
            "status": "Ready for transcription",
            "current_text": "",
            "history": []
        }
        self.is_ui_active = False
        self.output_queue = queue.Queue()
        self.ui_update_thread = None
        
        # Override the parent class message handlers with our enhanced versions
        self._setup_gradio_message_handlers()
        
    def _setup_gradio_message_handlers(self):
        """Set up enhanced message handlers for Gradio integration"""
        # Store references to the original handlers
        self._original_delta_handler = self.message_handlers.get("conversation.item.input_audio_transcription.delta")
        self._original_completed_handler = self.message_handlers.get("conversation.item.input_audio_transcription.completed")
        
        # Replace with our enhanced handlers
        self.message_handlers["conversation.item.input_audio_transcription.delta"] = self._enhanced_delta_handler
        self.message_handlers["conversation.item.input_audio_transcription.completed"] = self._enhanced_completed_handler
        
        # Add more detailed logging for debug purposes
        logger.debug(f"Enhanced message handlers set up, delta handler: {self._original_delta_handler is not None}, completed handler: {self._original_completed_handler is not None}")
        
    def _enhanced_delta_handler(self, msg):
        """Enhanced delta handler that updates the Gradio UI"""
        # Call the original handler if available
        if self._original_delta_handler:
            self._original_delta_handler(msg)
        
        # Update Gradio state
        delta = msg.get("delta", "")
        self.gradio_state["current_text"] += delta
        
        # Log the delta update
        logger.debug(f"Delta update received: '{delta}'")
        
        # Add to output queue for UI thread
        self.output_queue.put(("update_current", self.gradio_state["current_text"]))
    
    def _enhanced_completed_handler(self, msg):
        """Enhanced completed handler that updates the Gradio UI"""
        # Call the original handler if available
        if self._original_completed_handler:
            self._original_completed_handler(msg)
        
        # Get the transcript based on service type
        if self.service_type == "azure":
            try:
                transcript_raw = msg.get("transcript", "")
                transcript_json = json.loads(transcript_raw)
                transcript = transcript_json.get("text", "")
            except (json.JSONDecodeError, AttributeError):
                transcript = transcript_raw
        else:
            transcript = msg.get("transcript", "")
        
        # Log the completed transcript
        logger.debug(f"Completed transcript received: '{transcript}'")
        
        # Update Gradio state
        self.gradio_state["current_text"] = ""
        self.gradio_state["history"].append(transcript)
        
        # Add to output queue for UI thread
        self.output_queue.put(("update_completed", (transcript, self.gradio_state["history"])))
        
        logger.info(f"Completed transcript: {transcript}")
    
    def start_transcription(self) -> Tuple[bool, str]:
        """
        Start the transcription process with a UI update thread
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        # Reset state
        self.termination_event.clear()
        self.is_recording = True
        self.is_ui_active = True
        self.transcribed_text = []
        self.gradio_state = {
            "status": "Recording in progress...",
            "current_text": "",
            "history": []
        }
        
        # Clear output queue
        while not self.output_queue.empty():
            self.output_queue.get()
        
        # Start UI update thread
        self.ui_update_thread = threading.Thread(target=self._ui_update_worker)
        self.ui_update_thread.daemon = True
        self.ui_update_thread.start()
        
        # Start main transcription thread
        self.main_thread = threading.Thread(target=self._run_transcription)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        return True, "Transcription started"
    
    def _ui_update_worker(self):
        """Worker thread for processing UI updates"""
        try:
            logger.debug(f"UI update worker thread started, callback: {self.update_callback is not None}")
            while self.is_ui_active:
                try:
                    # Get the next update type with a short timeout
                    update_type, data = self.output_queue.get(timeout=0.5)
                    logger.debug(f"UI update received: {update_type}, data size: {len(str(data))}")
                    
                    # Process based on update type
                    if update_type == "update_current":
                        if self.update_callback:
                            current_text = data
                            history_text = "\n".join(self.gradio_state["history"])
                            logger.debug(f"Calling update_callback with current text: '{current_text[:30]}...'")
                            try:
                                # Save the callback to a temporary variable to ensure it's not None
                                callback = self.update_callback
                                if callback:
                                    # Direct call to the callback function
                                    callback("Status: 🎙️ Recording in progress...", current_text, history_text)
                            except Exception as e:
                                logger.error(f"Error in callback execution: {e}", exc_info=True)
                        else:
                            logger.warning("No update_callback available to update UI")
                    
                    elif update_type == "update_completed":
                        if self.update_callback:
                            transcript, history = data
                            history_text = "\n".join(history)
                            logger.debug(f"Calling update_callback with completed transcript: '{transcript[:30]}...'")
                            try:
                                # Save the callback to a temporary variable to ensure it's not None
                                callback = self.update_callback
                                if callback:
                                    # Direct call to the callback function
                                    callback("Status: 🎙️ Recording in progress...", "", history_text)
                            except Exception as e:
                                logger.error(f"Error in callback execution: {e}", exc_info=True)
                        else:
                            logger.warning("No update_callback available to update UI")
                    
                    elif update_type == "status":
                        if self.update_callback:
                            status = data
                            current_text = self.gradio_state.get("current_text", "")
                            history_text = "\n".join(self.gradio_state.get("history", []))
                            logger.debug(f"Calling update_callback with status: '{status}'")
                            try:
                                # Save the callback to a temporary variable to ensure it's not None
                                callback = self.update_callback
                                if callback:
                                    # Direct call to the callback function
                                    callback(status, current_text, history_text)
                            except Exception as e:
                                logger.error(f"Error in callback execution: {e}", exc_info=True)
                        else:
                            logger.warning("No update_callback available to update UI")
                    
                    # Mark as processed
                    self.output_queue.task_done()
                    
                except queue.Empty:
                    # No updates available, just continue
                    pass
                
                # Brief sleep to avoid CPU spin
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in UI update thread: {e}", exc_info=True)
        finally:
            logger.debug("UI update thread terminated")
    
    def _run_transcription(self):
        """Main thread to run the transcription process"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Set up the WebSocket connection task
            self.websocket_task = asyncio.ensure_future(self.setup_connection())
            
            # Wait for specified duration or until terminated
            start_time = time.time()
            while (time.time() - start_time < self.max_duration and 
                   not self.termination_event.is_set()):
                loop.run_until_complete(asyncio.sleep(0.1))
                
            # Log the reason for ending
            if self.termination_event.is_set():
                logger.info("Transcription terminated by user")
                self.output_queue.put(("status", "Status: ⏹️ Recording stopped by user"))
            else:
                logger.info("Transcription completed due to time limit")
                self.output_queue.put(("status", "Status: ⏹️ Recording reached time limit"))
                
            # Cancel the WebSocket task if still running
            if not self.websocket_task.done():
                self.websocket_task.cancel()
                try:
                    loop.run_until_complete(self.websocket_task)
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            logger.error(f"Error in main transcription thread: {e}")
            self.output_queue.put(("status", f"Status: ❌ Error: {str(e)}"))
        finally:
            # Ensure cleanup
            self.is_recording = False
            loop.close()
            logger.debug("Transcription thread terminated")
    
    def stop_transcription(self) -> Tuple[bool, str]:
        """
        Stop the transcription process
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        # If not recording, nothing to do
        if not self.is_recording:
            return False, "Not currently recording"
        
        # Signal termination
        self.termination_event.set()
        self.is_recording = False
        
        # Wait for threads to finish (with timeout)
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
        
        # Signal UI thread to stop
        self.is_ui_active = False
        if self.ui_update_thread and self.ui_update_thread.is_alive():
            self.ui_update_thread.join(timeout=2.0)
        
        return True, "Transcription stopped"
    
    def get_current_state(self) -> Tuple[str, str, str]:
        """
        Get the current state of the transcription
        
        Returns:
            Tuple[str, str, str]: Status, current text, history text
        """
        status = self.gradio_state.get("status", "Ready for transcription")
        if self.is_recording:
            status = "Status: 🎙️ Recording in progress..."
        else:
            status = "Status: Ready for transcription"
        
        current_text = self.gradio_state.get("current_text", "")
        history = self.gradio_state.get("history", [])
        history_text = "\n".join(history) if history else ""
        
        return f"Status: {status}", current_text, history_text
    
    def clear_history(self) -> None:
        """Clear the transcription history"""
        self.transcribed_text = []
        self.current_transcription = ""
        self.gradio_state = {
            "status": "Ready for transcription",
            "current_text": "",
            "history": []
        }


# Global variable to store the current service instance
# Don't create an instance at module level - only create when needed
gradio_transcription_service = None


def start_realtime_transcription(
    service_type: str = "azure",
    model: str = "gpt-4o-transcribe",
    noise_reduction: Optional[str] = None,
    turn_threshold: float = 0.5,
    include_logprobs: bool = True,
    max_duration: int = 60,
    update_callback=None
) -> Tuple[str, str, str]:
    """
    Start real-time transcription with the enhanced Gradio service
    
    Args:
        service_type: "azure" or "openai"
        model: Model name
        noise_reduction: Type of noise reduction
        turn_threshold: Voice activity detection threshold
        include_logprobs: Whether to include confidence scores
        max_duration: Maximum recording duration
        update_callback: Callback function for UI updates
        
    Returns:
        Tuple[str, str, str]: Status, current text, history text
    """
    global gradio_transcription_service
    
    # Validate parameters
    if service_type not in ["azure", "openai"]:
        return "Status: ❌ Invalid service type", "", ""
    
    if model not in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
        return "Status: ❌ Invalid model", "", ""
    
    if noise_reduction == "none":
        noise_reduction = None
    elif noise_reduction not in [None, "near_field", "far_field"]:
        return "Status: ❌ Invalid noise reduction setting", "", ""
    
    # Set up credentials
    kwargs = {}
    if service_type == "azure":
        # Check Azure credentials
        if not all([AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, AZURE_OPENAI_GPT4O_DEPLOYMENT_ID]):
            return "Status: ❌ Missing Azure OpenAI credentials", "", ""
        
        kwargs.update({
            "endpoint": AZURE_OPENAI_GPT4O_ENDPOINT,
            "deployment": AZURE_OPENAI_GPT4O_DEPLOYMENT_ID,
            "api_key": AZURE_OPENAI_GPT4O_API_KEY
        })
    else:
        # Check OpenAI credentials
        if not OPENAI_API_KEY:
            return "Status: ❌ Missing OpenAI API key", "", ""
        
        kwargs.update({"api_key": OPENAI_API_KEY})
    
    # Create a new service instance with updated parameters
    gradio_transcription_service = GradioTranscriptionService(
        service_type=service_type,
        model=model,
        noise_reduction=noise_reduction,
        turn_threshold=turn_threshold,
        include_logprobs=include_logprobs,
        max_duration=max_duration,
        # Here's the key fix: we need to store the update_callback correctly
        update_callback=update_callback,
        **kwargs
    )
    
    # Start the transcription
    success, message = gradio_transcription_service.start_transcription()
    
    if success:
        return "Status: 🎙️ Recording started. Speak into your microphone...", "", ""
    else:
        return f"Status: ❌ Failed to start recording: {message}", "", ""


def stop_realtime_transcription() -> Tuple[str, str, str]:
    """
    Stop the current real-time transcription
    
    Returns:
        Tuple[str, str, str]: Status, current text, history text
    """
    global gradio_transcription_service
    
    # If no service exists, nothing to stop
    if gradio_transcription_service is None:
        return "Status: ℹ️ No active transcription session", "", ""
    
    success, message = gradio_transcription_service.stop_transcription()
    
    if success:
        status, current, history = gradio_transcription_service.get_current_state()
        return "Status: ⏹️ Recording stopped", current, history
    else:
        return f"Status: ℹ️ {message}", "", ""


def get_realtime_transcription_status() -> Tuple[str, str, str]:
    """
    Get the current status of the transcription
    
    Returns:
        Tuple[str, str, str]: Status, current text, history text
    """
    global gradio_transcription_service
    
    # If no service exists, return default status
    if gradio_transcription_service is None:
        return "Status: Ready for transcription", "", ""
    
    return gradio_transcription_service.get_current_state()


def clear_realtime_transcription_history() -> Tuple[str, str, str]:
    """
    Clear the transcription history
    
    Returns:
        Tuple[str, str, str]: Status, empty current, empty history
    """
    global gradio_transcription_service
    
    # If no service exists, nothing to clear
    if gradio_transcription_service is None:
        return "Status: Ready for transcription", "", ""
    
    gradio_transcription_service.clear_history()
    return "Status: 🧹 Transcription history cleared", "", ""
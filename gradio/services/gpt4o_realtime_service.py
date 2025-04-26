"""
GPT-4o Real-time Transcription service for Azure Speech Recognition.
Implements WebSocket-based real-time transcription using OpenAI GPT-4o-transcribe model.
"""
import logging
import os
import threading
import asyncio
from typing import Dict, List, Tuple, Optional, Callable, Any
import sys
import time

# Add notebooks directory to path so we can import the transcription service
notebooks_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../notebooks"))
if notebooks_dir not in sys.path:
    sys.path.append(notebooks_dir)

from transcription_websocket_service import TranscriptionService, start_azure_transcription, start_openai_transcription

logger = logging.getLogger(__name__)

# GPT-4o-transcribe configuration from environment
AZURE_OPENAI_GPT4O_API_KEY = os.getenv("AZURE_OPENAI_GPT4O_API_KEY")
AZURE_OPENAI_GPT4O_ENDPOINT = os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT")
AZURE_OPENAI_GPT4O_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class GradioTranscriptionService:
    """
    Wrapper for TranscriptionService to work with Gradio interface.
    This class manages the state and interaction with the underlying WebSocket-based transcription.
    """
    
    def __init__(self):
        """Initialize the transcription service wrapper"""
        self.is_recording = False
        self.current_transcription = ""
        self.transcription_history = []
        self.transcription_service = None
        self.recording_thread = None
        self.update_callback = None
        self.current_status = "Status: Ready for real-time transcription"
    
    def start_transcription(
        self,
        service_type: str = "azure", 
        model: str = "gpt-4o-transcribe",
        noise_reduction: Optional[str] = None,
        turn_threshold: float = 0.5,
        turn_prefix_padding_ms: int = 300, 
        turn_silence_duration_ms: int = 500,
        include_logprobs: bool = True,
        max_duration: int = 300,  # 5 minutes max by default
        update_callback: Optional[Callable[[str, List[str]], None]] = None
    ) -> Tuple[bool, str]:
        """
        Start real-time transcription in a separate thread
        
        Args:
            service_type: Either "azure" or "openai"
            model: Model to use ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
            noise_reduction: Type of noise reduction (None, "near_field", or "far_field")
            turn_threshold: Voice activity detection threshold (0.0 to 1.0)
            turn_prefix_padding_ms: Padding time before speech detection (ms)
            turn_silence_duration_ms: Silent time to end a turn (ms)
            include_logprobs: Whether to include confidence scores
            max_duration: Maximum recording duration in seconds
            update_callback: Optional callback to update the UI with new transcription results
            
        Returns:
            Tuple[bool, str]: Success status and message
        """
        # Don't start if already recording
        if self.is_recording:
            return False, "Already recording"
        
        # Validate credentials based on service type
        if service_type == "azure":
            if not all([AZURE_OPENAI_GPT4O_API_KEY, AZURE_OPENAI_GPT4O_ENDPOINT, AZURE_OPENAI_GPT4O_DEPLOYMENT_ID]):
                return False, "Missing Azure OpenAI GPT-4o credentials"
        else:  # openai
            if not OPENAI_API_KEY:
                return False, "Missing OpenAI API key"
        
        # Reset state
        self.transcription_history = []
        self.current_transcription = ""
        self.update_callback = update_callback
        
        # Create service parameters
        kwargs = {
            "service_type": service_type,
            "model": model,
            "noise_reduction": noise_reduction,
            "turn_threshold": turn_threshold,
            "turn_prefix_padding_ms": turn_prefix_padding_ms,
            "turn_silence_duration_ms": turn_silence_duration_ms,
            "include_logprobs": include_logprobs
        }
        
        # Add service-specific credentials
        if service_type == "azure":
            kwargs.update({
                "endpoint": AZURE_OPENAI_GPT4O_ENDPOINT,
                "deployment": AZURE_OPENAI_GPT4O_DEPLOYMENT_ID,
                "api_key": AZURE_OPENAI_GPT4O_API_KEY
            })
        else:
            kwargs.update({
                "api_key": OPENAI_API_KEY
            })
        
        # Create the transcription service
        self.transcription_service = TranscriptionService(**kwargs)
        
        # Store a reference to the original handle_completed method to be used in our hook
        original_handle_completed = self.transcription_service._handle_completed
        
        # Create a hook for the handle_completed method to update UI real-time
        def handle_completed_hook(msg):
            # Call the original handler
            original_handle_completed(msg)
            
            # Get the transcript
            if service_type == "azure":
                try:
                    transcript_raw = msg.get("transcript", "")
                    transcript_json = json.loads(transcript_raw)
                    transcript = transcript_json.get("text", "")
                except (json.JSONDecodeError, AttributeError):
                    transcript = transcript_raw
            else:
                transcript = msg.get("transcript", "")
                
            # Update our view of the transcription
            self.current_transcription = transcript
            self.transcription_history = list(self.transcription_service.transcribed_text)
            
            # Log the update
            logger.debug(f"Updated transcription: {transcript}")
            
        # Replace the handler with our hooked version
        try:
            import json
            self.transcription_service._handle_completed = handle_completed_hook
        except Exception as e:
            logger.error(f"Failed to hook handler: {e}")
        
        # Helper function to run asyncio function in a thread-safe way
        def run_async_transcription():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                self.is_recording = True
                logger.debug("Starting transcription with max duration: %s seconds", max_duration)
                
                # Direct function call approach - using the appropriate function based on service type
                if service_type == "azure":
                    transcript, probs = start_azure_transcription(
                        endpoint=AZURE_OPENAI_GPT4O_ENDPOINT,
                        deployment=AZURE_OPENAI_GPT4O_DEPLOYMENT_ID,
                        api_key=AZURE_OPENAI_GPT4O_API_KEY,
                        duration=max_duration,
                        model=model,
                        noise_reduction=noise_reduction,
                        turn_threshold=turn_threshold,
                        include_logprobs=include_logprobs
                    )
                else:
                    transcript, probs = start_openai_transcription(
                        api_key=OPENAI_API_KEY,
                        duration=max_duration,
                        model=model,
                        noise_reduction=noise_reduction,
                        turn_threshold=turn_threshold,
                        include_logprobs=include_logprobs
                    )
                
                # Store the final transcript
                self.transcription_history = transcript if transcript else []
                
                # Ensure transcription stopped flag is set when done
                self.is_recording = False
                
                # Call update callback with final results if provided
                if self.update_callback:
                    self.update_callback("Status: ✅ Transcription complete", self.transcription_history)
                    
            except Exception as e:
                logger.error(f"Error in transcription thread: {e}")
                self.is_recording = False
                if self.update_callback:
                    self.update_callback(f"Status: ❌ Error: {str(e)}", [])
            finally:
                # Clean up the event loop
                loop.close()
        
        # Start the recording thread
        self.recording_thread = threading.Thread(target=run_async_transcription)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start a monitor thread to periodically update the UI
        def monitor_transcription():
            try:
                while self.is_recording:
                    if hasattr(self.transcription_service, 'transcribed_text'):
                        # Get current state
                        current = self.transcription_service.current_transcription
                        history = list(self.transcription_service.transcribed_text)
                        
                        # Update local state
                        self.current_transcription = current 
                        self.transcription_history = history
                        
                        logger.info(f"Monitor update - Current: {current[:30]}... History: {len(history)} items")
                    
                    # Sleep for a bit before checking again
                    time.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in monitor thread: {e}")
        
        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor_transcription)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return True, "Transcription started"
    
    def stop_transcription(self) -> Tuple[bool, str]:
        """
        Stop the current transcription
        
        Returns:
            Tuple[bool, str]: Success status and message
        """
        if not self.is_recording:
            return False, "Not currently recording"
        
        # Signal the transcription service to stop
        self.is_recording = False
        
        # Wait for the recording thread to finish (with timeout)
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
            
        return True, "Transcription stopped"
    
    def get_status(self) -> Tuple[str, str, List[str]]:
        """
        Get the current transcription status and results
        
        Returns:
            Tuple[str, str, List[str]]: Status message, current transcription, transcription history
        """
        status = "Status: 🎙️ Recording in progress..." if self.is_recording else "Status: Ready for transcription"
        
        if self.transcription_service and hasattr(self.transcription_service, 'current_transcription'):
            current = self.transcription_service.current_transcription
        else:
            current = ""
            
        return status, current, self.transcription_history
    
    def clear_history(self) -> None:
        """Clear the transcription history"""
        self.transcription_history = []
        self.current_transcription = ""


# Create a singleton instance of the service
realtime_transcription_service = GradioTranscriptionService()


def start_realtime_transcription(
    service_type: str = "azure",
    model: str = "gpt-4o-transcribe",
    noise_reduction: Optional[str] = None,
    turn_threshold: float = 0.5,
    include_logprobs: bool = True,
    max_duration: int = 60
) -> Tuple[str, str]:
    """
    Start real-time transcription using WebSockets with specified parameters.
    This function is designed to be called from the Gradio interface.
    
    Args:
        service_type: "azure" or "openai"
        model: "gpt-4o-transcribe" or "gpt-4o-mini-transcribe"
        noise_reduction: None, "near_field", or "far_field"
        turn_threshold: Voice activity detection threshold (0.0 to 1.0)
        include_logprobs: Whether to include confidence scores
        max_duration: Maximum recording duration in seconds
        
    Returns:
        Tuple[str, str]: Status message and empty string for initial transcription display
    """
    # Validate service choice
    if service_type not in ["azure", "openai"]:
        return "Status: ❌ Invalid service type. Choose 'azure' or 'openai'", ""
        
    # Validate model choice
    if model not in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
        return "Status: ❌ Invalid model. Choose 'gpt-4o-transcribe' or 'gpt-4o-mini-transcribe'", ""
    
    # Validate noise reduction
    if noise_reduction not in [None, "none", "near_field", "far_field"]:
        return "Status: ❌ Invalid noise reduction. Choose None, 'near_field', or 'far_field'", ""
        
    # Process noise reduction string
    if noise_reduction == "none":
        noise_reduction = None
        
    # Start the transcription
    success, message = realtime_transcription_service.start_transcription(
        service_type=service_type,
        model=model,
        noise_reduction=noise_reduction,
        turn_threshold=turn_threshold,
        include_logprobs=include_logprobs,
        max_duration=max_duration
    )
    
    if success:
        return "Status: 🎙️ Recording started. Speak into your microphone...", ""
    else:
        return f"Status: ❌ Failed to start recording: {message}", ""


def stop_realtime_transcription() -> Tuple[str, str, str]:
    """
    Stop the current real-time transcription
    
    Returns:
        Tuple[str, str, str]: Status message, current transcription text, history text
    """
    success, message = realtime_transcription_service.stop_transcription()
    
    # Get current status after stopping
    status, current, history = realtime_transcription_service.get_status()
    
    # Format history as a string
    history_text = "\n".join(history) if history else ""
    
    if success:
        return "Status: ⏹️ Recording stopped", current, history_text
    else:
        return f"Status: ℹ️ {message}", current, history_text


def get_realtime_transcription_status() -> Tuple[str, str, str]:
    """
    Get the current status of real-time transcription
    
    Returns:
        Tuple[str, str, str]: Status message, current transcription text, history text
    """
    status, current, history = realtime_transcription_service.get_status()
    
    # Format history as a string
    history_text = "\n".join(history) if history else ""
    
    return status, current, history_text


def clear_realtime_transcription_history() -> Tuple[str, str, str]:
    """
    Clear the real-time transcription history
    
    Returns:
        Tuple[str, str, str]: Status message, empty current transcription, empty history
    """
    realtime_transcription_service.clear_history()
    return "Status: 🧹 Transcription history cleared", "", ""
"""
Streaming Transcription Service.
Extends the base TranscriptionService with streaming capabilities for real-time UI updates.
"""
import os
import sys
import json
import asyncio
import time
import logging
from typing import Optional, Dict, Any, Callable, AsyncGenerator

# Add notebooks directory to path to import the base transcription service
notebooks_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../notebooks"))
if notebooks_dir not in sys.path:
    sys.path.append(notebooks_dir)

# Import the base TranscriptionService
from transcription_websocket_service import TranscriptionService

logger = logging.getLogger(__name__)


class StreamingTranscriptionService(TranscriptionService):
    """
    Extended TranscriptionService that yields streaming events for real-time UI updates.
    
    This class adds asynchronous streaming capabilities to the base TranscriptionService,
    making it suitable for integration with Gradio's async UI updates.
    """
    
    async def stream_transcription(self, duration=30, event_callback=None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time transcription with specified duration, yielding events as they occur

        Args:
            duration: Maximum recording duration in seconds
            event_callback: Optional callback function to process events in real-time:
                            callback(event_type, data) where event_type is one of:
                            "delta", "transcript", "status", "error"

        Yields:
            Dictionaries with transcription events containing:
            - event_type: "delta" (incremental update), "transcript" (completed), 
                        "status" (status update), or "error"
            - data: The content of the event (text for delta/transcript, message for status/error)
            - timestamp: When the event occurred
        """
        # Check if already recording
        if self.is_recording:
            yield {"event_type": "error", "data": "Already recording", "timestamp": time.time()}
            return

        # Clear audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        # Reset transcription state
        self.transcribed_text = []
        self.probs = []
        self.current_transcription = ""

        # Set recording flag
        self.is_recording = True

        # Start audio capture in a separate thread
        import threading
        audio_thread = threading.Thread(target=self._audio_capture)
        audio_thread.daemon = True
        audio_thread.start()

        # Yield initial status
        yield {
            "event_type": "status", 
            "data": f"Starting transcription for {duration} seconds",
            "timestamp": time.time()
        }

        # Set up the custom message handlers for streaming
        original_handlers = self.message_handlers.copy()
        
        # Create a queue for message passing between handlers and this coroutine
        message_queue = asyncio.Queue()
        
        # Define new handlers that both call the originals and add to the queue
        async def queue_delta(msg):
            delta = msg.get("delta", "")
            if original_handlers["conversation.item.input_audio_transcription.delta"]:
                original_handlers["conversation.item.input_audio_transcription.delta"](msg)
            
            event = {
                "event_type": "delta",
                "data": delta,
                "current_text": self.current_transcription,
                "timestamp": time.time()
            }
            await message_queue.put(event)
            if event_callback:
                event_callback("delta", delta)
        
        async def queue_completed(msg):
            if original_handlers["conversation.item.input_audio_transcription.completed"]:
                original_handlers["conversation.item.input_audio_transcription.completed"](msg)
            
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
            
            event = {
                "event_type": "transcript",
                "data": transcript,
                "transcript_history": list(self.transcribed_text),
                "timestamp": time.time()
            }
            await message_queue.put(event)
            if event_callback:
                event_callback("transcript", transcript)
        
        # Create handlers for other events
        async def queue_speech_started(msg):
            if "input_audio_buffer.speech_started" in original_handlers:
                original_handlers["input_audio_buffer.speech_started"](msg)
            event = {
                "event_type": "status",
                "data": "Speech detected, listening...",
                "timestamp": time.time()
            }
            await message_queue.put(event)
            if event_callback:
                event_callback("status", "Speech detected")
        
        async def queue_speech_stopped(msg):
            if "input_audio_buffer.speech_stopped" in original_handlers:
                original_handlers["input_audio_buffer.speech_stopped"](msg)
            event = {
                "event_type": "status",
                "data": "Speech stopped",
                "timestamp": time.time()
            }
            await message_queue.put(event)
            if event_callback:
                event_callback("status", "Speech stopped")
        
        async def queue_error(msg):
            if "error" in original_handlers:
                original_handlers["error"](msg)
            error_msg = msg.get("message", "Unknown error")
            event = {
                "event_type": "error",
                "data": error_msg,
                "timestamp": time.time()
            }
            await message_queue.put(event)
            if event_callback:
                event_callback("error", error_msg)
        
        # Update handlers with our async versions
        streaming_handlers = {
            "conversation.item.input_audio_transcription.delta": queue_delta,
            "conversation.item.input_audio_transcription.completed": queue_completed,
            "input_audio_buffer.speech_started": queue_speech_started,
            "input_audio_buffer.speech_stopped": queue_speech_stopped,
            "error": queue_error
        }
        
        # Create a wrapper for websocket receive that uses our streaming handlers
        async def streaming_receive_messages(websocket):
            """Modified receive_messages that uses streaming handlers"""
            try:
                while True:
                    try:
                        message = await websocket.recv()
                        try:
                            msg = json.loads(message)
                            msg_type = msg.get("type")
                            
                            # Call the appropriate streaming handler based on message type
                            if msg_type in streaming_handlers:
                                await streaming_handlers[msg_type](msg)
                            else:
                                # For other message types, just pass to original handler
                                handler = original_handlers.get(
                                    msg_type, 
                                    lambda m: print(f"ℹ️ Message type: {m.get('type')}", flush=True)
                                )
                                if callable(handler):
                                    handler(msg)
                                
                                # Also queue status messages for certain events
                                if msg_type in ["transcription_session.created", "transcription_session.updated"]:
                                    event = {
                                        "event_type": "status",
                                        "data": f"{msg_type.replace('_', ' ').title()}",
                                        "timestamp": time.time()
                                    }
                                    await message_queue.put(event)
                                    if event_callback:
                                        event_callback("status", event["data"])
                                        
                        except json.JSONDecodeError:
                            print(f"Received non-JSON message: {message}", flush=True)
                            
                    except websockets.exceptions.ConnectionClosedError:
                        print("\n🔌 WebSocket connection closed", flush=True)
                        event = {
                            "event_type": "status",
                            "data": "WebSocket connection closed",
                            "timestamp": time.time()
                        }
                        await message_queue.put(event)
                        if event_callback:
                            event_callback("status", "Connection closed")
                        break
                        
            except Exception as e:
                print(f"\n❌ Error in receive_messages: {e}")
                event = {
                    "event_type": "error",
                    "data": f"Error in receive_messages: {e}",
                    "timestamp": time.time()
                }
                await message_queue.put(event)
                if event_callback:
                    event_callback("error", str(e))
            finally:
                print("📥 Message receiving complete")
                event = {
                    "event_type": "status",
                    "data": "Message receiving complete",
                    "timestamp": time.time()
                }
                await message_queue.put(event)
                if event_callback:
                    event_callback("status", "Message receiving complete")
        
        try:
            # Determine the appropriate connection details based on service type
            if self.service_type == "azure":
                # Headers for authentication
                headers = {"api-key": self.azure_api_key}
                # WebSocket URL for Azure OpenAI
                ws_url = f"wss://{self.azure_endpoint}/openai/realtime?intent=transcription&deployment={self.azure_deployment}&api-version=2024-10-01-preview"
            else:
                # Default to OpenAI
                # WebSocket URL for OpenAI Realtime API
                ws_url = "wss://api.openai.com/v1/realtime?intent=transcription"
                # Headers for authentication
                import websockets
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "OpenAI-Beta": "realtime=v1",  # Required beta header
                }
            
            # Connect to WebSocket
            service_name = "Azure OpenAI" if self.service_type == "azure" else "OpenAI"
            print(f"🔄 Connecting to {service_name} Realtime API...")
            
            # Create tasks for processing the queue while we're running the WebSocket
            async def process_message_queue():
                start_time = time.time()
                last_time_update = 0
                try:
                    while self.is_recording and (time.time() - start_time < duration):
                        try:
                            # Get the next message with a timeout
                            event = await asyncio.wait_for(message_queue.get(), timeout=0.1)
                            
                            # Add time remaining information to status events
                            if event["event_type"] == "status":
                                time_elapsed = time.time() - start_time
                                time_remaining = max(0, duration - time_elapsed)
                                event["time_remaining"] = round(time_remaining)
                                
                            # Yield the event
                            yield event
                            # Mark the task as done
                            message_queue.task_done()
                        except asyncio.TimeoutError:
                            # No message available, send periodic time updates (every second)
                            current_time = int(time.time())
                            if current_time > last_time_update:
                                last_time_update = current_time
                                time_elapsed = time.time() - start_time
                                time_remaining = max(0, duration - time_elapsed)
                                yield {
                                    "event_type": "status",
                                    "data": f"Recording in progress. Time remaining: {round(time_remaining)} seconds",
                                    "time_remaining": round(time_remaining),
                                    "timestamp": time.time()
                                }
                except asyncio.CancelledError:
                    print("Message queue processing cancelled")
                except GeneratorExit:
                    print("🛑 Message queue generator exit requested")
                    # Properly handle GeneratorExit by not suppressing it
                    raise
                finally:
                    print("Message queue processing completed")
            
            # Create a termination flag
            termination_requested = asyncio.Event()
            
            # Create a task to automatically end the transcription after duration
            async def auto_terminate():
                try:
                    await asyncio.sleep(duration)
                    termination_requested.set()
                except asyncio.CancelledError:
                    pass
            
            # Start the termination timer
            termination_timer = asyncio.create_task(auto_terminate())
            
            import websockets
            async with websockets.connect(ws_url, additional_headers=headers) as websocket:
                print("🔗 WebSocket connection established")
                yield {
                    "event_type": "status",
                    "data": "WebSocket connection established",
                    "timestamp": time.time()
                }
                
                # Send session configuration
                await self.send_session_update(websocket)
                yield {
                    "event_type": "status",
                    "data": "Session configuration sent",
                    "timestamp": time.time()
                }
                
                # Start tasks for audio sending, message receiving, and queue processing
                audio_task = asyncio.create_task(self.send_audio(websocket))
                receiver_task = asyncio.create_task(streaming_receive_messages(websocket))
                
                # Process messages from the queue and yield them
                queue_processor = process_message_queue()
                try:
                    async for event in queue_processor:
                        yield event
                        
                        # Check if termination has been requested
                        if termination_requested.is_set():
                            break
                except GeneratorExit:
                    # Handle graceful shutdown when the generator is closed
                    print("🛑 Generator exit requested, shutting down gracefully")
                    # We'll continue to the cleanup in the finally block
                    pass
                    
                # Cancel the tasks
                audio_task.cancel()
                receiver_task.cancel()
                try:
                    await asyncio.gather(audio_task, receiver_task, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
                
        except websockets.exceptions.InvalidStatus as e:
            error_msg = f"Invalid status: {e}"
            print(f"❌ {error_msg}")
            yield {"event_type": "error", "data": error_msg, "timestamp": time.time()}
        except websockets.exceptions.ConnectionClosedError as e:
            error_msg = f"Connection closed unexpectedly: {e}"
            print(f"❌ {error_msg}")
            yield {"event_type": "error", "data": error_msg, "timestamp": time.time()}
        except Exception as e:
            error_msg = f"WebSocket connection error: {e}"
            print(f"❌ {error_msg}")
            yield {"event_type": "error", "data": error_msg, "timestamp": time.time()}
        finally:
            # Stop recording
            self.is_recording = False
            print("✅ Transcription session ended")
            # Cancel termination timer if it's still running
            if not termination_timer.done():
                termination_timer.cancel()
            # Yield final status
            yield {
                "event_type": "status",
                "data": "Transcription session ended",
                "timestamp": time.time()
            }
            if event_callback:
                event_callback("status", "Transcription session ended")


async def async_stream_transcription(
    service_type="azure",
    endpoint=None,
    deployment=None,
    api_key=None,
    model="gpt-4o-transcribe",
    noise_reduction=None,
    turn_threshold=0.5,
    turn_prefix_padding_ms=300,
    turn_silence_duration_ms=500,
    include_logprobs=True,
    duration=30,
    event_callback=None
):
    """
    Convenience function to stream transcription events asynchronously
    
    Args:
        service_type: "azure" or "openai"
        endpoint: Azure OpenAI endpoint (for Azure)
        deployment: Azure OpenAI deployment ID (for Azure)
        api_key: API key
        model: Model to use
        noise_reduction: Type of noise reduction
        turn_threshold: Voice activity detection threshold
        turn_prefix_padding_ms: Padding time before speech detection
        turn_silence_duration_ms: Silent time to end a turn
        include_logprobs: Whether to include confidence scores
        duration: Maximum recording duration
        event_callback: Optional callback function to process events in real-time
        
    Yields:
        Dictionaries with transcription events
    """
    # Create the appropriate service
    if service_type == "azure":
        service = StreamingTranscriptionService(
            service_type="azure",
            model=model,
            noise_reduction=noise_reduction,
            turn_threshold=turn_threshold,
            turn_prefix_padding_ms=turn_prefix_padding_ms,
            turn_silence_duration_ms=turn_silence_duration_ms,
            include_logprobs=include_logprobs,
            endpoint=endpoint or os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT"),
            deployment=deployment or os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID"),
            api_key=api_key or os.getenv("AZURE_OPENAI_GPT4O_API_KEY")
        )
    else:
        service = StreamingTranscriptionService(
            service_type="openai",
            model=model,
            noise_reduction=noise_reduction,
            turn_threshold=turn_threshold,
            turn_prefix_padding_ms=turn_prefix_padding_ms,
            turn_silence_duration_ms=turn_silence_duration_ms,
            include_logprobs=include_logprobs,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    
    # Stream the transcription
    async for event in service.stream_transcription(duration=duration, event_callback=event_callback):
        yield event
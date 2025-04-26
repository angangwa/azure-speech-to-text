"""Speech Transcription with OpenAI Realtime API.
Demonstrates how to use the OpenAI or Azure OpenAI API for live speech transcription using WebSockets.

For the new gpt-4o-transcribe and gpt-4o-mini-transcribe models, that are currently in preview.
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import queue
import os
import threading
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()


class TranscriptionService:
    """Class for handling real-time speech transcription via WebSockets.

    This class provides a unified interface for both OpenAI and Azure OpenAI
    transcription services using WebSockets and asyncio.
    """

    def __init__(
        self,
        service_type: str = "openai",
        model: str = "gpt-4o-transcribe",
        noise_reduction: Optional[str] = None,
        turn_threshold: float = 0.5,
        turn_prefix_padding_ms: int = 300,
        turn_silence_duration_ms: int = 500,
        include_logprobs: bool = True,
        **kwargs,
    ):
        """Initialize the transcription service.

        Args:
            service_type: Either "openai" or "azure"
            model: Model to use ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
            noise_reduction: Type of noise reduction (None, "near_field", or "far_field")
            turn_threshold: Voice activity detection threshold (0.0 to 1.0)
            turn_prefix_padding_ms: Padding time before speech detection (ms)
            turn_silence_duration_ms: Silent time to end a turn (ms)
            include_logprobs: Whether to include confidence scores in results
            **kwargs: Service-specific parameters:
                For Azure: endpoint, deployment, api_key
                For OpenAI: api_key
                If not provided, will check environment variables.
        """
        self.service_type = service_type.lower()
        self.service_params = kwargs

        # Validate model
        if model not in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
            raise ValueError(
                "Model must be either 'gpt-4o-transcribe' or 'gpt-4o-mini-transcribe'"
            )
        self.model = model

        # Validate noise reduction
        if noise_reduction not in [None, "near_field", "far_field"]:
            raise ValueError(
                "Noise reduction must be None, 'near_field', or 'far_field'"
            )
        self.noise_reduction = noise_reduction

        # VAD settings
        self.turn_threshold = turn_threshold
        self.turn_prefix_padding_ms = turn_prefix_padding_ms
        self.turn_silence_duration_ms = turn_silence_duration_ms
        self.include_logprobs = include_logprobs

        # Set up session configuration based on init parameters
        self.session_config = self._build_session_config()

        # Validate and set up credentials
        self._setup_credentials()

        # Audio parameters
        self.format = pyaudio.paInt16  # 16-bit PCM (pcm16)
        self.channels = 1  # Mono audio
        self.rate = 24000  # 24kHz as recommended by OpenAI
        self.chunk = 1024  # Number of frames per buffer

        # State variables
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcribed_text = []
        self.probs = []

        # Configure message handlers
        self._setup_message_handlers()
        
    def _setup_credentials(self):
        """Set up and validate credentials based on service type"""
        if self.service_type == "azure":
            # Try to get Azure credentials from kwargs, then environment variables
            self.azure_endpoint = self.service_params.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_deployment = self.service_params.get("deployment") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            self.azure_api_key = self.service_params.get("api_key") or os.getenv("AZURE_OPENAI_KEY")
            
            # Validate that all required Azure credentials are present
            if not all([self.azure_endpoint, self.azure_deployment, self.azure_api_key]):
                missing = []
                if not self.azure_endpoint:
                    missing.append("endpoint")
                if not self.azure_deployment:
                    missing.append("deployment")
                if not self.azure_api_key:
                    missing.append("api_key")
                raise ValueError(
                    f"Missing required Azure OpenAI credentials: {', '.join(missing)}. "
                    f"Provide them as parameters or set the corresponding environment variables."
                )
        else:
            # For OpenAI, get API key from kwargs or environment
            self.openai_api_key = self.service_params.get("api_key") or os.getenv("OPENAI_API_KEY")
            
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )

    def _build_session_config(self):
        """Build session configuration based on initialization parameters"""
        # Set up noise reduction configuration
        if self.noise_reduction is not None:
            noise_reduction_config = {"type": self.noise_reduction}
        else:
            noise_reduction_config = None

        # Create base config
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": self.model},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.turn_threshold,
                    "prefix_padding_ms": self.turn_prefix_padding_ms,
                    "silence_duration_ms": self.turn_silence_duration_ms,
                },
                "input_audio_noise_reduction": noise_reduction_config,
            },
        }

        # Add logprobs if enabled
        if self.include_logprobs:
            config["session"]["include"] = ["item.input_audio_transcription.logprobs"]

        return config

    def _setup_message_handlers(self):
        """Set up handlers for different WebSocket message types"""
        self.message_handlers = {
            "conversation.item.input_audio_transcription.delta": self._handle_delta,
            "conversation.item.input_audio_transcription.completed": self._handle_completed,
            "transcription_session.created": lambda msg: print(
                "✅ Transcription session created", flush=True
            ),
            "transcription_session.updated": lambda msg: print(
                "✅ Transcription session updated", flush=True
            ),
            "input_audio_buffer.speech_started": lambda msg: print(
                "\n🎤 Speech detected, listening...", flush=True
            ),
            "input_audio_buffer.speech_stopped": lambda msg: print(
                "🔇 Speech stopped", flush=True
            ),
            "conversation.item.created": lambda msg: print(
                "📝 New conversation item created", flush=True
            ),
            "error": lambda msg: print(f"\n❌ Error: {msg.get('message')}", flush=True),
            # input_audio_buffer.committed
            "input_audio_buffer.committed": lambda msg: print(
                "📤 Audio buffer committed", flush=True
            ),
        }

        # Current state for incremental updates
        self.current_transcription = ""

    def _handle_delta(self, msg):
        """Handle incremental transcription updates"""
        delta = msg.get("delta", "")
        self.current_transcription += delta

    def _handle_completed(self, msg):
        """Handle completed transcription"""
        transcript_raw = msg.get("transcript", "")

        # Handle different formats between OpenAI and Azure
        if self.service_type == "azure":
            try:
                # Azure returns a JSON string with the transcript in a "text" field
                # Example: '{\n  "text": "Now I\'m going to speak again."\n}'
                transcript_json = json.loads(transcript_raw)
                transcript = transcript_json.get("text", "")
                print(f'\n📝 Azure Completed Transcript: "{transcript}"', flush=True)
            except json.JSONDecodeError:
                # Fallback if the JSON parsing fails
                print(f"\n⚠️ Could not parse Azure transcript JSON: {transcript_raw}")
                transcript = transcript_raw
        else:
            # OpenAI returns the transcript directly as a string
            transcript = transcript_raw
            print(f'\n📝 Completed Transcript: "{transcript}"', flush=True)

        self.transcribed_text.append(transcript)
        self.probs.append(msg.get("logprobs", {}))
        self.current_transcription = ""

    def _audio_capture(self):
        """Capture audio from microphone and add to queue"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        print("🎙️ Recording started...")

        try:
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.audio_queue.put(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("🎙️ Recording stopped")

    def get_session_config(self, model="gpt-4o-transcribe", include_logprobs=True):
        """Return standard session configuration with options"""
        config = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": {"model": model},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "input_audio_noise_reduction": None,
            },
        }

        if include_logprobs:
            config["session"]["include"] = ["item.input_audio_transcription.logprobs"]

        return config

    async def send_session_update(self, websocket):
        """Send session configuration to the API"""
        await websocket.send(json.dumps(self.session_config))
        print("✅ Sent session configuration")
        return True

    async def send_audio(self, websocket):
        """Send audio data from queue to WebSocket"""
        try:
            while self.is_recording or not self.audio_queue.empty():
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    # Encode audio data as base64
                    encoded_data = base64.b64encode(audio_data).decode("utf-8")

                    # Create audio buffer message
                    message = {
                        "type": "input_audio_buffer.append",
                        "audio": encoded_data,
                    }

                    try:
                        await websocket.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosedError:
                        print("❌ WebSocket connection closed while sending data")
                        break
                    except Exception as e:
                        print(f"❌ Error sending audio data: {e}")
                        break

                # Short sleep to avoid CPU hogging
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"❌ Error in send_audio: {e}")
        finally:
            print("📤 Audio sending complete")

    async def receive_messages(self, websocket):
        """Receive and process messages from the WebSocket"""
        try:
            while True:
                try:
                    message = await websocket.recv()
                    try:
                        msg = json.loads(message)
                        msg_type = msg.get("type")

                        # Call the appropriate handler based on message type
                        handler = self.message_handlers.get(
                            msg_type,
                            lambda m: print(
                                f"ℹ️ Message type: {m.get('type')}", flush=True
                            ),
                        )
                        handler(msg)

                    except json.JSONDecodeError:
                        print(f"Received non-JSON message: {message}", flush=True)

                except websockets.exceptions.ConnectionClosedError:
                    print("\n🔌 WebSocket connection closed", flush=True)
                    break

        except Exception as e:
            print(f"\n❌ Error in receive_messages: {e}")
        finally:
            print("📥 Message receiving complete")

    async def setup_connection(self):
        """Set up the WebSocket connection to the appropriate service"""
        # Reset transcription state
        self.transcribed_text = []
        self.probs = []
        self.current_transcription = ""

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
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "OpenAI-Beta": "realtime=v1",  # Required beta header
            }

        try:
            # Connect to WebSocket
            service_name = "Azure OpenAI" if self.service_type == "azure" else "OpenAI"
            print(f"🔄 Connecting to {service_name} Realtime API...")

            async with websockets.connect(
                ws_url, additional_headers=headers
            ) as websocket:
                print("🔗 WebSocket connection established")

                # Send session configuration
                await self.send_session_update(websocket)

                # Start tasks for sending audio and receiving messages
                await asyncio.gather(
                    self.send_audio(websocket), self.receive_messages(websocket)
                )
        except websockets.exceptions.InvalidStatus as e:
            print(f"❌ Invalid status: {e}.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"❌ Connection closed unexpectedly: {e}")
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
        finally:
            self.is_recording = False
            print("✅ WebSocket connection closed")

    def start_transcription(self, duration=30):
        """Start real-time transcription with specified duration

        Args:
            duration: Recording duration in seconds

        Returns:
            Tuple of (transcribed_text, probability_data)
        """
        # Check if already recording
        if self.is_recording:
            print("⚠️ Already recording. Please wait for the current session to finish.")
            return None, None

        # Clear audio queue
        while not self.audio_queue.empty():
            self.audio_queue.get()

        # Set recording flag
        self.is_recording = True

        # Start audio capture in a separate thread
        audio_thread = threading.Thread(target=self._audio_capture)
        audio_thread.daemon = True
        audio_thread.start()

        # Start WebSocket connection in asyncio event loop
        print(
            f"🚀 Starting transcription for {duration} seconds. Speak into your microphone."
        )
        
        # Try to get the current event loop, create a new one if it doesn't exist or is closed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # "There is no current event loop in thread" error
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run the WebSocket task asynchronously
        websocket_task = asyncio.ensure_future(self.setup_connection())

        try:
            # Wait for specified duration
            loop.run_until_complete(asyncio.sleep(duration))
        except KeyboardInterrupt:
            print("⛔ Interrupted by user")
        finally:
            # Clean up
            self.is_recording = False

            # Give time for final audio to be sent and processed
            loop.run_until_complete(asyncio.sleep(2))

            # Cancel WebSocket task if still running
            if not websocket_task.done():
                websocket_task.cancel()
                try:
                    loop.run_until_complete(websocket_task)
                except asyncio.CancelledError:
                    pass

            print("✅ Transcription session ended")

            # Return the full transcript
            return self.transcribed_text, self.probs

    def display_transcript(self):
        """Display the full transcript"""
        if self.transcribed_text:
            print("📋 Full Transcript:")
            print("-------------------")
            print("\n".join(self.transcribed_text))
        else:
            print("No transcript available. Run the transcription function first.")


def start_openai_transcription(
    duration=30,
    model="gpt-4o-transcribe",
    noise_reduction=None,
    turn_threshold=0.5,
    turn_prefix_padding_ms=300,
    turn_silence_duration_ms=500,
    include_logprobs=True,
    api_key=None,
):
    """Simplified function to start OpenAI transcription

    Args:
        duration: Recording duration in seconds
        model: Model to use ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
        noise_reduction: Type of noise reduction (None, "near_field", or "far_field")
        turn_threshold: Voice activity detection threshold (0.0 to 1.0)
        turn_prefix_padding_ms: Padding time before speech detection (ms)
        turn_silence_duration_ms: Silent time to end a turn (ms)
        include_logprobs: Whether to include confidence scores in results
        api_key: OpenAI API key (optional, falls back to OPENAI_API_KEY env variable)
    """
    service = TranscriptionService(
        service_type="openai",
        model=model,
        noise_reduction=noise_reduction,
        turn_threshold=turn_threshold,
        turn_prefix_padding_ms=turn_prefix_padding_ms,
        turn_silence_duration_ms=turn_silence_duration_ms,
        include_logprobs=include_logprobs,
        api_key=api_key,
    )
    transcript, probs = service.start_transcription(duration=duration)
    service.display_transcript()
    return transcript, probs


def start_azure_transcription(
    endpoint=None,
    deployment=None,
    api_key=None,
    duration=30,
    model="gpt-4o-transcribe",
    noise_reduction=None,
    turn_threshold=0.5,
    turn_prefix_padding_ms=300,
    turn_silence_duration_ms=500,
    include_logprobs=True,
):
    """Simplified function to start Azure OpenAI transcription

    Args:
        endpoint: Azure OpenAI endpoint URL (optional, falls back to AZURE_OPENAI_ENDPOINT env variable)
        deployment: Azure OpenAI deployment name (optional, falls back to AZURE_OPENAI_DEPLOYMENT env variable)
        api_key: Azure OpenAI API key (optional, falls back to AZURE_OPENAI_KEY env variable)
        duration: Recording duration in seconds
        model: Model to use ("gpt-4o-transcribe" or "gpt-4o-mini-transcribe")
        noise_reduction: Type of noise reduction (None, "near_field", or "far_field")
        turn_threshold: Voice activity detection threshold (0.0 to 1.0)
        turn_prefix_padding_ms: Padding time before speech detection (ms)
        turn_silence_duration_ms: Silent time to end a turn (ms)
        include_logprobs: Whether to include confidence scores in results
    """
    service = TranscriptionService(
        service_type="azure",
        model=model,
        noise_reduction=noise_reduction,
        turn_threshold=turn_threshold,
        turn_prefix_padding_ms=turn_prefix_padding_ms,
        turn_silence_duration_ms=turn_silence_duration_ms,
        include_logprobs=include_logprobs,
        endpoint=endpoint,
        deployment=deployment,
        api_key=api_key,
    )
    transcript, probs = service.start_transcription(duration=duration)
    service.display_transcript()
    return transcript, probs


if __name__ == "__main__":
    # Example usage
    # Run using Azure OpenAI
    # Run using OpenAI
    # api_key = os.getenv("OPENAI_API_KEY")
    # transcript, probs = start_openai_transcription(duration=60)

    # Azure OpenAI Azure
    # endpoint = "<>.openai.azure.com"
    # deployment = "gpt-4o-transcribe"
    # api_key = "<>"

    endpoint = os.environ.get("AZURE_OPENAI_GPT4O_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID")
    api_key = os.environ.get("AZURE_OPENAI_GPT4O_API_KEY")
    print(endpoint, deployment, api_key)

    transcript, probs = start_azure_transcription(
        endpoint=endpoint, deployment=deployment, api_key=api_key, duration=60
    )

"""
Core Azure Speech Recognition service implementation.
Handles continuous speech recognition from microphone and audio files.
"""
import logging
import threading
import azure.cognitiveservices.speech as speechsdk
from typing import Callable, Dict, Optional, Tuple, Any

from config import create_speech_config

logger = logging.getLogger(__name__)


class SpeechRecognitionService:
    """Service class for Azure Speech Recognition functionality"""

    def __init__(self):
        """Initialize Speech Recognition Service"""
        self.speech_config = create_speech_config()

        # Recognition state
        self.recognizing_text = ""
        self.recognized_history = ""
        self.is_listening = False
        self.is_stopping = False  # New flag to track stopping state
        self.is_file_processing = False
        self.recognizer = None
        self.file_recognizer = None

        # Diarization settings
        self.use_diarization = False
        self.conversation_transcriber = None
        self.file_conversation_transcriber = None

        # For tracking file processing
        self.file_audio_length = None
        self.file_session_stopped = False

        # Thread safety
        self.update_lock = threading.Lock()

    def recognizing_callback(self, evt):
        """Callback for intermediate recognition results"""
        text = evt.result.text
        speaker_id = getattr(evt.result, "speaker_id", None)

        if speaker_id and self.use_diarization:
            logger.debug(f"RECOGNIZING (Speaker {speaker_id}): {text}")
            with self.update_lock:
                self.recognizing_text = f"Speaker {speaker_id}: {text}"
        else:
            logger.debug(f"RECOGNIZING: {text}")
            with self.update_lock:
                self.recognizing_text = text

    def recognized_callback(self, evt):
        """Callback for final recognition results"""
        text = evt.result.text
        speaker_id = getattr(evt.result, "speaker_id", None)

        if speaker_id and self.use_diarization:
            logger.debug(f"RECOGNIZED (Speaker {speaker_id}): {text}")
            if text.strip():
                with self.update_lock:
                    self.recognized_history += f"Speaker {speaker_id}: {text}\n"
                    self.recognizing_text = ""
        else:
            logger.debug(f"RECOGNIZED: {text}")
            if text.strip():
                with self.update_lock:
                    self.recognized_history += text + "\n"
                    self.recognizing_text = ""

    def session_started_callback(self, evt):
        """Callback for session started events"""
        logger.debug(f"SESSION STARTED")

    def session_stopped_callback(self, evt):
        """Callback for session stopped events"""
        logger.debug(f"SESSION STOPPED")
        # Reset stopping state when session is actually stopped
        if self.is_stopping:
            logger.info("Recognition is now fully stopped")
            with self.update_lock:
                self.is_listening = False
                self.is_stopping = False

    def speech_start_detected_callback(self, evt):
        """Callback for speech start detection"""
        logger.debug(f"SPEECH START DETECTED")

    def speech_end_detected_callback(self, evt):
        """Callback for speech end detection"""
        logger.debug(f"SPEECH END DETECTED")

    def file_processing_completed_callback(self, evt):
        """Callback for file processing completion"""
        logger.debug(f"File processing completed or stopped: {evt}")
        # Don't immediately set is_file_processing to False
        # We'll wait a bit longer to ensure all recognition results are received
        self.file_session_stopped = True

    def connect_callbacks(self, recognizer):
        """Connect all callbacks to the recognizer"""
        recognizer.recognizing.connect(self.recognizing_callback)
        recognizer.recognized.connect(self.recognized_callback)
        recognizer.session_started.connect(self.session_started_callback)
        recognizer.session_stopped.connect(self.session_stopped_callback)
        recognizer.speech_start_detected.connect(self.speech_start_detected_callback)
        recognizer.speech_end_detected.connect(self.speech_end_detected_callback)
        recognizer.canceled.connect(
            self.session_stopped_callback
        )  # Add canceled handler

    def connect_file_callbacks(self, recognizer):
        """Connect all callbacks for file recognition"""
        self.connect_callbacks(recognizer)
        recognizer.canceled.connect(self.file_processing_completed_callback)
        recognizer.session_stopped.connect(self.file_processing_completed_callback)

    def configure_diarization(self, enable: bool):
        """
        Configure diarization settings

        Args:
            enable (bool): Whether to enable diarization
        """
        self.use_diarization = enable
        logger.info(f"Diarization settings updated: enabled={enable}")

    def setup_speech_config(self):
        """Set up speech config with current diarization settings"""
        if self.use_diarization:
            # Configure the speech config for diarization
            self.speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption,
                value="TrueText",
            )
            self.speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
                value="true",
            )
            logger.debug(f"Speech config configured for diarization")
        return self.speech_config

    def start_microphone_recognition(self) -> bool:
        """
        Start the speech recognition process from microphone

        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if self.is_listening or self.is_stopping:
                logger.debug("Already listening or stopping")
                return False

            logger.debug("Creating audio config for microphone")
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

            # Apply diarization settings to speech config
            self.setup_speech_config()

            logger.debug("Creating recognizer")
            if self.use_diarization:
                # Use ConversationTranscriber for diarization
                logger.debug("Using ConversationTranscriber for diarization")
                self.conversation_transcriber = (
                    speechsdk.transcription.ConversationTranscriber(
                        speech_config=self.speech_config, audio_config=audio_config
                    )
                )
                # Connect callbacks
                self.conversation_transcriber.transcribing.connect(
                    self.recognizing_callback
                )
                self.conversation_transcriber.transcribed.connect(
                    self.recognized_callback
                )
                self.conversation_transcriber.session_stopped.connect(
                    self.session_stopped_callback
                )
                self.conversation_transcriber.canceled.connect(
                    self.session_stopped_callback
                )

                # Start transcription
                self.conversation_transcriber.start_transcribing_async()
            else:
                # Use standard SpeechRecognizer
                self.recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, audio_config=audio_config
                )
                # Connect all callbacks
                logger.debug("Connecting callbacks")
                self.connect_callbacks(self.recognizer)
                # Start continuous recognition
                logger.debug("Starting continuous recognition")
                self.recognizer.start_continuous_recognition_async()

            self.is_listening = True
            logger.info(
                f"Recognition started successfully with diarization={self.use_diarization}"
            )

            return True
        except Exception as e:
            logger.error(f"Error starting recognition: {e}")
            return False

    def stop_microphone_recognition(self) -> bool:
        """
        Stop the speech recognition process from microphone

        Returns:
            bool: True if stopping initiated successfully, False otherwise
        """
        try:
            if not self.is_listening or self.is_stopping:
                logger.debug("Not currently listening or already stopping")
                return False

            logger.debug("Initiating stop of continuous recognition")
            # Set stopping flag before making the async call
            self.is_stopping = True

            if self.use_diarization and self.conversation_transcriber:
                self.conversation_transcriber.stop_transcribing_async()
                # Don't set to None yet - we need it for callbacks
            elif self.recognizer:
                self.recognizer.stop_continuous_recognition_async()
                # Don't set to None yet - we need it for callbacks

            logger.info("Recognition stop initiated successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping recognition: {e}")
            self.is_listening = False
            self.is_stopping = False
            return False

    def get_recognition_status(self) -> Tuple[str, str, str]:
        """
        Get the current recognition status and text

        Returns:
            Tuple[str, str, str]: Status message, current recognizing text, history
        """
        with self.update_lock:
            current_recognizing = self.recognizing_text
            current_history = self.recognized_history
            is_listening_now = self.is_listening
            is_stopping_now = self.is_stopping

        if is_stopping_now:
            status = "Status: ⏳ Stopping recognition..."
        elif is_listening_now:
            status = "Status: 🎙️ Listening"
        else:
            status = "Status: 🔇 Not listening"
        logger.debug(
            f"Status: {status}, Recognizing: '{current_recognizing}', History length: {len(current_history)}"
        )

        return status, current_recognizing, current_history

    def clear_history(self) -> None:
        """Clear the recognition history"""
        logger.info("Clearing history")
        with self.update_lock:
            self.recognized_history = ""
            self.recognizing_text = ""

    def start_file_recognition(self, file_path: str) -> bool:
        """
        Start speech recognition from a file

        Args:
            file_path (str): Path to the audio file

        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Clear previous recognition data
            with self.update_lock:
                self.recognizing_text = ""
                self.recognized_history = ""

            # Reset file processing state
            self.file_session_stopped = False

            logger.debug(f"Creating audio config for file: {file_path}")
            audio_config = speechsdk.audio.AudioConfig(filename=file_path)

            # Apply diarization settings to speech config
            self.setup_speech_config()

            logger.debug("Creating file recognizer")
            if self.use_diarization:
                # Use ConversationTranscriber for diarization
                logger.debug("Using ConversationTranscriber for file diarization")
                self.file_conversation_transcriber = (
                    speechsdk.transcription.ConversationTranscriber(
                        speech_config=self.speech_config, audio_config=audio_config
                    )
                )

                # Connect callbacks
                self.file_conversation_transcriber.transcribing.connect(
                    self.recognizing_callback
                )
                self.file_conversation_transcriber.transcribed.connect(
                    self.recognized_callback
                )
                self.file_conversation_transcriber.session_stopped.connect(
                    self.file_processing_completed_callback
                )
                self.file_conversation_transcriber.canceled.connect(
                    self.file_processing_completed_callback
                )

                # Start transcription
                logger.info("Starting file transcription with diarization")
                self.is_file_processing = True
                self.file_conversation_transcriber.start_transcribing_async()
            else:
                # Use standard SpeechRecognizer
                self.file_recognizer = speechsdk.SpeechRecognizer(
                    speech_config=self.speech_config, audio_config=audio_config
                )

                # Connect all callbacks
                logger.debug("Connecting file recognition callbacks")
                self.connect_file_callbacks(self.file_recognizer)

                # Start continuous recognition
                logger.info("Starting file recognition")
                self.is_file_processing = True
                self.file_recognizer.start_continuous_recognition()

            return True
        except Exception as e:
            logger.error(f"Error starting file recognition: {e}")
            self.is_file_processing = False
            return False

    def stop_file_recognition(self) -> bool:
        """
        Stop the file speech recognition process

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            if not self.is_file_processing:
                logger.info("No file is currently being processed")
                return False

            logger.info("Stopping file recognition")
            if self.use_diarization and self.file_conversation_transcriber:
                self.file_conversation_transcriber.stop_transcribing_async()
                self.file_conversation_transcriber = None
            elif self.file_recognizer:
                self.file_recognizer.stop_continuous_recognition()
                self.file_recognizer = None

            # Mark as stopped by user
            self.is_file_processing = False
            self.file_session_stopped = True

            logger.info("File recognition stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping file recognition: {e}")
            self.is_file_processing = False
            self.file_session_stopped = True
            return False

    def get_file_processing_status(self) -> str:
        """
        Get the current file processing status

        Returns:
            str: Status message
        """
        if self.is_file_processing:
            diarization_info = " with diarization" if self.use_diarization else ""
            if self.file_audio_length:
                return f"Status: 📄 Processing file{diarization_info}... (Audio length: {self.file_audio_length:.2f} seconds)"
            return f"Status: 📄 Processing file{diarization_info}..."
        else:
            diarization_info = " with diarization" if self.use_diarization else ""
            if self.file_audio_length:
                return f"Status: ✅ File processing{diarization_info} complete (Audio length: {self.file_audio_length:.2f} seconds)"
            return f"Status: ✅ File processing{diarization_info} complete"


# Create a singleton instance
speech_service = SpeechRecognitionService()

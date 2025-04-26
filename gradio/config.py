"""
Configuration module for Azure Speech Recognition application.
Handles loading environment variables and configuring services.
"""
import os
import logging
import argparse
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create root logger reference for level control
root_logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Azure Speech Service configuration
SPEECH_KEY = os.getenv("SPEECH_KEY")
SERVICE_REGION = os.getenv("SERVICE_REGION")

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "whisper")

# Azure OpenAI GPT-4o configuration
AZURE_OPENAI_GPT4O_API_KEY = os.getenv("AZURE_OPENAI_GPT4O_API_KEY")
AZURE_OPENAI_GPT4O_ENDPOINT = os.getenv("AZURE_OPENAI_GPT4O_ENDPOINT")
AZURE_OPENAI_GPT4O_DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_ID")

# Direct OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Create Azure Speech config
def create_speech_config():
    """Create and return a speech config object using environment variables"""
    if not SPEECH_KEY or not SERVICE_REGION:
        logger.error("Missing SPEECH_KEY or SERVICE_REGION environment variables.")
        raise ValueError(
            "Missing Azure Speech credentials. Please check your .env file."
        )

    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY, region=SERVICE_REGION
    )
    speech_config.speech_recognition_language = "en-US"
    return speech_config


def set_logging_level(debug_mode=False):
    """Set the logging level based on debug mode"""
    level = logging.DEBUG if debug_mode else logging.INFO
    root_logger.setLevel(level)
    logger.debug(f"Logging level set to: {'DEBUG' if debug_mode else 'INFO'}")
    return level


def get_current_logging_level():
    """Get the current logging level name"""
    return logging.getLevelName(root_logger.level)


# Verify configurations
def verify_configs():
    """Verify that all required configurations are present"""
    configs = {
        "Speech Service": bool(SPEECH_KEY and SERVICE_REGION),
        "Azure OpenAI": bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT),
        "Azure OpenAI GPT-4o": bool(AZURE_OPENAI_GPT4O_API_KEY and AZURE_OPENAI_GPT4O_ENDPOINT and AZURE_OPENAI_GPT4O_DEPLOYMENT_ID),
        "OpenAI API": bool(OPENAI_API_KEY)
    }

    for name, available in configs.items():
        status = "available" if available else "NOT available"
        logger.info(f"{name} configuration is {status}")

    return configs

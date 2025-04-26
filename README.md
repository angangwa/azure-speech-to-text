# Speech to text on Azure

## Overview

This repository provides a comprehensive demonstration of multiple speech-to-text transcription approaches available on Microsoft Azure. It showcases various methods for converting spoken language to written text, from Speech Service SDK functionality to advanced AI models. The project includes:

- Working examples of all major Azure speech transcription services
- Interactive demos using Gradio for easy testing without coding
- Jupyter notebooks with detailed code examples and explanations
- Utility functions for real-world speech processing scenarios

Ideal for developers looking to implement speech-to-text functionality in their applications, data scientists working with audio data, or anyone interested in exploring Azure's speech recognition capabilities.

## Features

1. Real-time speech to text from microphone and wav file
2. Fast Transcription using Azure AI Speech Service REST API
3. OpenAI Whisper model via Azure OpenAI
4. Speaker diarization (identifying different speakers in conversations)
5. Azure OpenAI GPT-4o-transcribe model for high-accuracy transcription
6. **NEW** WebSocket-based real-time transcription with OpenAI Realtime API
7. Streaming transcription for both completed audio files and live audio

![screenshot](./diagrams/screenshot.png)

## Quick Start (commands for git bash on Windows)

### 1. Create `.env` file locally

Create a new file `.env` with API keys and region details for Azure speech-to-text services.

```.env
# Azure Speech Service credentials (required for basic speech recognition, diarization, and fast transcription)
SPEECH_KEY=<your_azure_speech_service_key>
SERVICE_REGION=<your_azure_region> # e.g. uksouth

# Azure OpenAI Service credentials (required for Whisper model)
AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT_ID=whisper

# Azure OpenAI GPT-4o credentials (required for GPT-4o-transcribe model)
AZURE_OPENAI_GPT4O_API_KEY=<your_azure_openai_gpt4o_api_key>
AZURE_OPENAI_GPT4O_ENDPOINT=<your_azure_openai_gpt4o_endpoint>
AZURE_OPENAI_GPT4O_DEPLOYMENT_ID=<your_azure_openai_gpt4o_deployment_id>

# Direct OpenAI API credentials (optional, for using OpenAI's services directly)
OPENAI_API_KEY=<your_openai_api_key>
```

> **IMPORTANT NOTE:** The endpoint URL formats for Azure OpenAI services differ based on the service:
> 
> **For Whisper model (AZURE_OPENAI_ENDPOINT):**
> - Requires the **full endpoint URL** including deployment path and API version
> - Format: `https://<resource-name>.cognitiveservices.azure.com/openai/deployments/<deployment-id>/audio/translations?api-version=2024-06-01`
> - Example: `https://myopenai.cognitiveservices.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01`
> 
> **For GPT-4o-transcribe model (AZURE_OPENAI_GPT4O_ENDPOINT):**
> - Requires **only the domain name** without https:// or any paths
> - Format: `<resource-name>.openai.azure.com`
> - Example: `myopenai.openai.azure.com`
>
> The code handles these different formats appropriately - for GPT-4o-transcribe, the domain is used to construct the proper WebSocket URL for real-time transcription.

### Prerequisites on Azure

Before using this repository, you'll need to set up the following Azure resources:

1. **Azure AI Speech Service**
   - Create an Azure AI Speech resource in the Azure portal
   - Note the key and region for your `SPEECH_KEY` and `SERVICE_REGION` variables
   - Required for: Basic speech recognition, diarization, and fast transcription

2. **Azure OpenAI Service for Whisper**
   - Create an Azure OpenAI resource
   - Deploy the Whisper model with your chosen deployment name
   - Note the endpoint and API key for your `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` variables
   - Set `AZURE_OPENAI_DEPLOYMENT_ID` to your deployment name (e.g., "whisper")
   - Required for: OpenAI Whisper transcription

3. **Azure OpenAI Service for GPT-4o-transcribe**
   - Create an Azure OpenAI resource (can be the same as for Whisper)
   - Deploy the GPT-4o model and enable transcription capability
   - Note the endpoint URI including the deployment path for `AZURE_OPENAI_GPT4O_ENDPOINT`
   - Note the API key for `AZURE_OPENAI_GPT4O_API_KEY`
   - Set `AZURE_OPENAI_GPT4O_DEPLOYMENT_ID` to your deployment name
   - Required for: GPT-4o-transcribe and real-time WebSocket transcription

4. **OpenAI API (Optional)**
   - Only needed if you want to use OpenAI services directly instead of through Azure
   - Get an API key from OpenAI and set it as `OPENAI_API_KEY`

Each component of this repository can be used independently based on which Azure resources you have available.

### 2. Install

`python -m venv .venv`

`source .venv/Scripts/activate`

`pip install -r requirements.txt -r requirements-dev.txt`

### 3. Notebook

Review [azure-speech-to-text.ipynb](./notebooks/azure-speech-to-text.ipynb).

Or skip directly to next part.

### 4. Run Demo

`python gradio/app.py`

### 5. Play with the demo on Browser

[http://127.0.0.1:7860](http://127.0.0.1:7860)

### 6. Debug Mode

To run the application in debug mode, set the `DEBUG` environment variable to `True`:

```
export DEBUG=True
```

## Contributing (Committing changes)

Install pre-commit for basic checks and fixes before commit.

`pre-commit install`

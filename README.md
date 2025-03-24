# Speech to text on Azure

## Quick Start (commands for git bash on Windows)

### 1. Create `.env` file locally

Create a new file `.env` with API keys and region details from Azure AI Speech to Text service.

```.env
SPEECH_KEY=<>
SERVICE_REGION=<e.g. uksouth>
# For OpenAI Whisper model
AZURE_OPENAI_API_KEY=<>
AZURE_OPENAI_ENDPOINT=<>
AZURE_OPENAI_DEPLOYMENT_ID=whisper
```

### 2. Install

`python -m venv .venv`

`source .venv/Scripts/activate`

`pip install -r requirements.txt -r requirements-dev.txt`

### 3. Notebook

Review [speech-to-text-apis.ipynb](./notebooks/speech-to-text-apis.ipynb).

Or skip directly to next part.

### 4. Run Demo

`python gradio\app.py`

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
"""
Speech-to-Text using OpenAI audio transcription API (gpt-4o-transcribe, whisper-1).
No local model loading — inference runs on OpenAI infrastructure.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI_CLIENT = True
except ImportError:
    HAS_OPENAI_CLIENT = False
    OpenAI = None  # type: ignore


class OpenAISpeechRecognizer:
    """Transcribes audio via OpenAI audio API. No local model weights."""

    DEFAULT_MODEL = "gpt-4o-transcribe"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if not HAS_OPENAI_CLIENT:
            raise ImportError("openai library required: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self._client = OpenAI(api_key=self.api_key)
        print(f"OpenAI ASR initialized (model: {self.model})")

    def transcribe(
        self,
        audio_path: str,
        language: str = "de",
        task: str = "transcribe",
        verbose: bool = False,
    ) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language or "de",
                response_format="text",
            )
        # response_format="text" returns the transcript string directly
        return str(response).strip()


def get_openai_recognizer(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[OpenAISpeechRecognizer]:
    """Return an OpenAISpeechRecognizer, or None if unavailable (missing package or key)."""
    if not HAS_OPENAI_CLIENT:
        return None

    try:
        import config
        _key = api_key or getattr(config, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
        _model = model or getattr(config, "OPENAI_ASR_MODEL", OpenAISpeechRecognizer.DEFAULT_MODEL)
    except ImportError:
        _key = api_key or os.environ.get("OPENAI_API_KEY")
        _model = model or OpenAISpeechRecognizer.DEFAULT_MODEL

    if not _key:
        return None

    try:
        return OpenAISpeechRecognizer(api_key=_key, model=_model)
    except Exception as e:
        print(f"Warning: OpenAI ASR init failed: {e}")
        return None

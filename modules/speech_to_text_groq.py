"""
Speech-to-Text using Groq's hosted Whisper API (no local model weights).
"""

import os
from pathlib import Path
from typing import Optional

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    Groq = None  # type: ignore


class GroqSpeechRecognizer:
    """Transcribes audio via Groq's Whisper API. No local model loading."""

    DEFAULT_MODEL = "whisper-large-v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if not HAS_GROQ:
            raise ImportError("groq library required: pip install groq")

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self._client = Groq(api_key=self.api_key)
        print(f"Groq ASR initialized (model: {self.model})")

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
                file=(path.name, f.read()),
                model=self.model,
                language=language or "de",
                response_format="json",
            )
        return response.text.strip()


def get_groq_recognizer(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[GroqSpeechRecognizer]:
    """Return a GroqSpeechRecognizer, or None if unavailable (missing package or key)."""
    if not HAS_GROQ:
        return None

    try:
        import config
        _key = api_key or getattr(config, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
        _model = model or getattr(config, "GROQ_MODEL", GroqSpeechRecognizer.DEFAULT_MODEL)
    except ImportError:
        _key = api_key or os.environ.get("GROQ_API_KEY")
        _model = model or GroqSpeechRecognizer.DEFAULT_MODEL

    if not _key:
        return None

    try:
        return GroqSpeechRecognizer(api_key=_key, model=_model)
    except Exception as e:
        print(f"Warning: Groq ASR init failed: {e}")
        return None

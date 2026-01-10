"""
Speech-to-Text module using OpenAI Whisper for transcribing audio to text.
"""

import torch
from pathlib import Path
from typing import Optional

# Try to import whisper
try:
    import whisper
    HAS_WHISPER = True
except ImportError as e:
    HAS_WHISPER = False
    whisper = None


class SpeechToTextRecognizer:
    """Speech-to-Text recognizer using OpenAI Whisper model."""
    
    def __init__(self, model_size: str = "medium", device: Optional[str] = None):
        """
        Initialize Whisper model.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        if not HAS_WHISPER:
            raise ImportError(
                "whisper library is required. Install with: pip install openai-whisper"
            )
        
        self.model_size = model_size
        self.device = device or self._get_device()
        self.model = None
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Auto-detect device."""
        if torch.cuda.is_available():
            return "cuda"
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            print(f"Loading Whisper model '{self.model_size}' on device '{self.device}'...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{self.model_size}': {e}")
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "de",
        task: str = "transcribe",
        verbose: bool = False
    ) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'de' for German, 'en' for English)
                      If None, auto-detect language
            task: Task type ('transcribe' or 'translate')
            verbose: Whether to print progress information
            
        Returns:
            Transcribed text string
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language=language if language else None,
                task=task,
                verbose=verbose
            )
            
            # Extract text
            transcribed_text = result.get("text", "").strip()
            
            return transcribed_text
            
        except Exception as e:
            # Check if error is NaN-related on MPS device
            error_str = str(e)
            # Check for NaN errors - can be ValueError or RuntimeError with NaN in message
            is_nan_error = (
                "nan" in error_str.lower() or 
                "invalid values" in error_str.lower() or
                "expected parameter logits" in error_str.lower()
            )
            is_mps_device = self.device == "mps"
            
            # Fallback to CPU if NaN error on MPS
            if is_nan_error and is_mps_device:
                print(f"Warning: Whisper failed on MPS with NaN error, falling back to CPU...")
                # Reload model on CPU
                self.device = "cpu"
                self._load_model()
                # Retry transcription on CPU
                result = self.model.transcribe(
                    audio_path,
                    language=language if language else None,
                    task=task,
                    verbose=verbose
                )
                transcribed_text = result.get("text", "").strip()
                return transcribed_text
            else:
                # Not a NaN error on MPS, or not on MPS - raise original error
                raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def transcribe_with_details(
        self,
        audio_path: str,
        language: str = "de",
        task: str = "transcribe",
        verbose: bool = False
    ) -> dict:
        """
        Transcribe audio file and return full result with details.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'de' for German)
            task: Task type ('transcribe' or 'translate')
            verbose: Whether to print progress information
            
        Returns:
            Full result dictionary from Whisper with:
            - text: Transcribed text
            - language: Detected language
            - segments: List of segments with timestamps
            - etc.
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            result = self.model.transcribe(
                audio_path,
                language=language if language else None,
                task=task,
                verbose=verbose
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")


# Global instance
_speech_recognizer = None


def get_speech_recognizer(
    model_size: Optional[str] = None,
    device: Optional[str] = None
) -> Optional[SpeechToTextRecognizer]:
    """
    Get or create global speech recognizer instance.
    
    Args:
        model_size: Whisper model size. If None, uses config default.
        device: Device to use. If None, uses config default or auto-detect.
        
    Returns:
        SpeechToTextRecognizer instance, or None if whisper is not available
    """
    global _speech_recognizer
    
    if not HAS_WHISPER:
        return None
    
    if _speech_recognizer is None:
        # Try to get defaults from config
        if model_size is None or device is None:
            try:
                import config
                if model_size is None:
                    model_size = getattr(config, 'ASR_MODEL', 'medium')
                if device is None:
                    device = getattr(config, 'ASR_DEVICE', None)
            except ImportError:
                pass
        
        if model_size is None:
            model_size = "medium"
        
        try:
            _speech_recognizer = SpeechToTextRecognizer(
                model_size=model_size,
                device=device
            )
        except Exception as e:
            print(f"Warning: Failed to initialize speech recognizer: {e}")
            return None
    
    return _speech_recognizer

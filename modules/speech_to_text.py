"""
Speech-to-Text module using OpenAI Whisper for transcribing audio to text.
Supports both Whisper and macOS native Speech framework.
"""

import os
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

# Try to import macOS Speech framework
try:
    from modules.speech_to_text_macos import (
        get_macos_speech_recognizer,
        HAS_MACOS_SPEECH as HAS_MACOS_SPEECH_IMPORT
    )
    HAS_MACOS_SPEECH = HAS_MACOS_SPEECH_IMPORT
except ImportError:
    HAS_MACOS_SPEECH = False

# Try to import Groq ASR backend
try:
    from modules.speech_to_text_groq import get_groq_recognizer, HAS_GROQ
except ImportError:
    HAS_GROQ = False
    get_groq_recognizer = None  # type: ignore

# Try to import OpenAI ASR backend
try:
    from modules.speech_to_text_openai import get_openai_recognizer, HAS_OPENAI_CLIENT
except ImportError:
    HAS_OPENAI_CLIENT = False
    get_openai_recognizer = None  # type: ignore


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
        """Auto-detect device.
        
        Note: Whisper has known issues with MPS (NaN errors in Categorical distribution),
        so we use CPU on Apple Silicon for Whisper. MPS can be used for other models.
        """
        if torch.cuda.is_available():
            return "cuda"
        # Whisper has known NaN issues on MPS, so use CPU on Apple Silicon
        # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        #     return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load Whisper model with fallback to CPU if MPS fails."""
        try:
            print(f"Loading Whisper model '{self.model_size}' on device '{self.device}'...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Whisper model '{self.model_size}' loaded successfully on {self.device}")
        except Exception as e:
            # If MPS fails, try falling back to CPU
            if self.device == "mps":
                print(f"Warning: Failed to load Whisper on MPS: {e}")
                print("Falling back to CPU...")
                self.device = "cpu"
                try:
                    self.model = whisper.load_model(self.model_size, device="cpu")
                    print(f"Whisper model '{self.model_size}' loaded successfully on CPU (fallback)")
                except Exception as cpu_error:
                    raise RuntimeError(
                        f"Failed to load Whisper model '{self.model_size}' on both MPS and CPU. "
                        f"MPS error: {e}, CPU error: {cpu_error}"
                    )
            else:
                # For non-MPS devices, raise the original error
                raise RuntimeError(f"Failed to load Whisper model '{self.model_size}' on {self.device}: {e}")
    
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
    device: Optional[str] = None,
    engine: Optional[str] = None
) -> Optional[SpeechToTextRecognizer]:
    """
    Get or create global speech recognizer instance.
    Supports both Whisper and macOS Speech framework.
    
    Args:
        model_size: Whisper model size. If None, uses config default.
        device: Device to use. If None, uses config default or auto-detect.
        engine: "whisper" or "macos". If None, uses config default.
        
    Returns:
        SpeechToTextRecognizer instance (or macOS recognizer wrapper), or None if not available
    """
    global _speech_recognizer
    
    # Try to get engine from config
    if engine is None:
        try:
            import config
            engine = getattr(config, 'ASR_ENGINE', 'whisper')
        except ImportError:
            engine = 'whisper'
    
    # Use macOS Speech if requested
    if engine == "macos":
        
        # Check if macOS Speech is available
        if not HAS_MACOS_SPEECH:
            print("Warning: macOS Speech framework is not available (not macOS or pyobjc-framework-Speech not installed)")
            print("Falling back to Whisper...")
            engine = 'whisper'  # Fallback to Whisper
        else:
            # Try to initialize macOS Speech
            try:
                import config
                language = getattr(config, 'ASR_LANGUAGE', 'de')
                # Convert language code for macOS (e.g., 'de' -> 'de-DE')
                if language == 'de':
                    language = 'de-DE'
                elif language == 'en':
                    language = 'en-US'
                else:
                    language = f"{language}-{language.upper()}"
                
                
                macos_recognizer = get_macos_speech_recognizer(language=language)
                if macos_recognizer:
                    # Create a wrapper to match Whisper interface
                    class MacOSRecognizerWrapper:
                        def __init__(self, recognizer):
                            self.recognizer = recognizer
                            self._whisper_fallback = None
                        
                        def transcribe(self, audio_path, language=None, task=None, verbose=False):
                            
                            # Convert language code if needed
                            if language == 'de':
                                language = 'de-DE'
                            elif language == 'en':
                                language = 'en-US'
                            
                            try:
                                result = self.recognizer.transcribe(audio_path, language, task, verbose)
                                return result
                            except RuntimeError as e:
                                # Check if it's a timeout error
                                if "timeout" in str(e).lower():
                                    print(f"Warning: macOS Speech timed out, falling back to Whisper...")
                                    # Fallback to Whisper
                                    if self._whisper_fallback is None:
                                        # Initialize Whisper fallback
                                        if HAS_WHISPER:
                                            import config
                                            model_size = getattr(config, 'ASR_MODEL', 'medium')
                                            device = getattr(config, 'ASR_DEVICE', None)
                                            try:
                                                self._whisper_fallback = SpeechToTextRecognizer(model_size=model_size, device=device)
                                            except Exception as init_error:
                                                raise RuntimeError(f"macOS Speech timed out and Whisper initialization failed: {init_error}")
                                        else:
                                            raise RuntimeError("macOS Speech timed out and Whisper is not available")
                                    
                                    
                                    # Use Whisper for transcription
                                    try:
                                        whisper_result = self._whisper_fallback.transcribe(audio_path, language=whisper_language, task=task, verbose=verbose)
                                        return whisper_result
                                    except Exception as whisper_error:
                                        raise RuntimeError(f"macOS Speech timed out and Whisper transcription failed: {whisper_error}")
                                else:
                                    raise
                            except Exception as e:
                                raise
                    
                    return MacOSRecognizerWrapper(macos_recognizer)
                else:
                    print("Warning: Failed to initialize macOS speech recognizer")
                    print("Falling back to Whisper...")
                    engine = 'whisper'  # Fallback to Whisper
            except Exception as e:
                print(f"Warning: Failed to initialize macOS speech recognizer: {e}")
                print("Falling back to Whisper...")
                engine = 'whisper'  # Fallback to Whisper
    
    # Use OpenAI cloud ASR (gpt-4o-transcribe etc.)
    if engine == "openai":
        rec = get_openai_recognizer() if get_openai_recognizer is not None else None
        if rec is not None:
            return rec
        print("Warning: OpenAI ASR unavailable (missing package or OPENAI_API_KEY), falling back to Whisper")
        engine = "whisper"

    # Use Groq cloud ASR
    if engine == "groq":
        rec = get_groq_recognizer() if get_groq_recognizer is not None else None
        if rec is not None:
            return rec
        print("Warning: Groq ASR unavailable (missing package or GROQ_API_KEY), falling back to Whisper")
        engine = "whisper"

    # Use Whisper (default or fallback)
    if engine == "whisper":
        
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
    
    return None

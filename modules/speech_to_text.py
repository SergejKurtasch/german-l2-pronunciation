"""
Speech-to-Text module using OpenAI Whisper for transcribing audio to text.
Supports both Whisper and macOS native Speech framework.
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

# Try to import macOS Speech framework
try:
    from modules.speech_to_text_macos import (
        get_macos_speech_recognizer,
        HAS_MACOS_SPEECH as HAS_MACOS_SPEECH_IMPORT
    )
    HAS_MACOS_SPEECH = HAS_MACOS_SPEECH_IMPORT
except ImportError:
    HAS_MACOS_SPEECH = False


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
        # #region agent log
        import json, time
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_requested","message":"macOS Speech requested","data":{"has_macos_speech":HAS_MACOS_SPEECH},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        # Check if macOS Speech is available
        if not HAS_MACOS_SPEECH:
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_not_available","message":"macOS Speech not available, falling back to Whisper","data":{},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
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
                
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_init_start","message":"Initializing macOS Speech","data":{"language":language},"timestamp":int(time.time()*1000)})+'\n')
                # #endregion
                
                macos_recognizer = get_macos_speech_recognizer(language=language)
                if macos_recognizer:
                    # Create a wrapper to match Whisper interface
                    class MacOSRecognizerWrapper:
                        def __init__(self, recognizer):
                            self.recognizer = recognizer
                            self._whisper_fallback = None
                        
                        def transcribe(self, audio_path, language=None, task=None, verbose=False):
                            # #region agent log
                            import json, time
                            transcribe_start = time.time()
                            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:start","message":"macOS Speech transcribe called","data":{"audio_path":str(audio_path),"language":language},"timestamp":int(time.time()*1000)})+'\n')
                            # #endregion
                            
                            # Convert language code if needed
                            if language == 'de':
                                language = 'de-DE'
                            elif language == 'en':
                                language = 'en-US'
                            
                            try:
                                result = self.recognizer.transcribe(audio_path, language, task, verbose)
                                transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                # #region agent log
                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:completed","message":"macOS Speech transcribe completed","data":{"result_length":len(result) if result else 0,"result_preview":result[:50] if result else None},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                # #endregion
                                return result
                            except RuntimeError as e:
                                # Check if it's a timeout error
                                if "timeout" in str(e).lower():
                                    # #region agent log
                                    transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:timeout_fallback","message":"macOS Speech timeout, falling back to Whisper","data":{"error":str(e)},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                    # #endregion
                                    print(f"Warning: macOS Speech timed out, falling back to Whisper...")
                                    # Fallback to Whisper
                                    if self._whisper_fallback is None:
                                        # #region agent log
                                        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_init_start","message":"Initializing Whisper fallback","data":{"has_whisper":HAS_WHISPER},"timestamp":int(time.time()*1000)})+'\n')
                                        # #endregion
                                        # Initialize Whisper fallback
                                        if HAS_WHISPER:
                                            import config
                                            model_size = getattr(config, 'ASR_MODEL', 'medium')
                                            device = getattr(config, 'ASR_DEVICE', None)
                                            # #region agent log
                                            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                                f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_init_params","message":"Whisper init parameters","data":{"model_size":model_size,"device":device},"timestamp":int(time.time()*1000)})+'\n')
                                            # #endregion
                                            try:
                                                self._whisper_fallback = SpeechToTextRecognizer(model_size=model_size, device=device)
                                                # #region agent log
                                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_init_completed","message":"Whisper fallback initialized","data":{},"timestamp":int(time.time()*1000)})+'\n')
                                                # #endregion
                                            except Exception as init_error:
                                                # #region agent log
                                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_init_error","message":"Whisper init error","data":{"error":str(init_error),"error_type":type(init_error).__name__},"timestamp":int(time.time()*1000)})+'\n')
                                                # #endregion
                                                raise RuntimeError(f"macOS Speech timed out and Whisper initialization failed: {init_error}")
                                        else:
                                            raise RuntimeError("macOS Speech timed out and Whisper is not available")
                                    
                                    # #region agent log
                                    whisper_transcribe_start = time.time()
                                    # Convert language code for Whisper (e.g., 'de-DE' -> 'de', 'en-US' -> 'en')
                                    whisper_language = language
                                    if whisper_language:
                                        if whisper_language.lower().startswith('de'):
                                            whisper_language = 'de'
                                        elif whisper_language.lower().startswith('en'):
                                            whisper_language = 'en'
                                        else:
                                            # Extract base language code (first 2 chars before hyphen)
                                            whisper_language = whisper_language.split('-')[0].lower()
                                    else:
                                        whisper_language = 'de'
                                    
                                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_transcribe_start","message":"Starting Whisper transcription","data":{"audio_path":str(audio_path),"original_language":language,"whisper_language":whisper_language},"timestamp":int(time.time()*1000)})+'\n')
                                    # #endregion
                                    
                                    # Use Whisper for transcription
                                    try:
                                        whisper_result = self._whisper_fallback.transcribe(audio_path, language=whisper_language, task=task, verbose=verbose)
                                        whisper_transcribe_elapsed = (time.time() - whisper_transcribe_start) * 1000
                                        transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                        # #region agent log
                                        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_fallback_completed","message":"Whisper fallback completed","data":{"result_length":len(whisper_result) if whisper_result else 0,"result_preview":whisper_result[:100] if whisper_result else None,"whisper_elapsed_ms":int(whisper_transcribe_elapsed)},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                        # #endregion
                                        return whisper_result
                                    except Exception as whisper_error:
                                        whisper_transcribe_elapsed = (time.time() - whisper_transcribe_start) * 1000
                                        transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                        # #region agent log
                                        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                            f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:whisper_transcribe_error","message":"Whisper transcription error","data":{"error":str(whisper_error),"error_type":type(whisper_error).__name__,"whisper_elapsed_ms":int(whisper_transcribe_elapsed)},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                        # #endregion
                                        raise RuntimeError(f"macOS Speech timed out and Whisper transcription failed: {whisper_error}")
                                else:
                                    # Other runtime error - re-raise
                                    transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                    # #region agent log
                                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                        f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:exception","message":"macOS Speech transcribe exception","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                    # #endregion
                                    raise
                            except Exception as e:
                                transcribe_elapsed = (time.time() - transcribe_start) * 1000
                                # #region agent log
                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-transcribe","hypothesisId":"D","location":"modules/speech_to_text.py:MacOSRecognizerWrapper:transcribe:exception","message":"macOS Speech transcribe exception","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000),"elapsed_ms":int(transcribe_elapsed)})+'\n')
                                # #endregion
                                raise
                    
                    # #region agent log
                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_success","message":"macOS Speech initialized successfully","data":{},"timestamp":int(time.time()*1000)})+'\n')
                    # #endregion
                    return MacOSRecognizerWrapper(macos_recognizer)
                else:
                    # #region agent log
                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_init_failed","message":"macOS Speech initialization returned None, falling back to Whisper","data":{},"timestamp":int(time.time()*1000)})+'\n')
                    # #endregion
                    print("Warning: Failed to initialize macOS speech recognizer")
                    print("Falling back to Whisper...")
                    engine = 'whisper'  # Fallback to Whisper
            except Exception as e:
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:macos_exception","message":"macOS Speech initialization exception, falling back to Whisper","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)})+'\n')
                # #endregion
                print(f"Warning: Failed to initialize macOS speech recognizer: {e}")
                print("Falling back to Whisper...")
                engine = 'whisper'  # Fallback to Whisper
    
    # Use Whisper (default or fallback)
    if engine == "whisper":
        # #region agent log
        import json, time
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:whisper_selected","message":"Using Whisper engine","data":{"has_whisper":HAS_WHISPER,"model_size":model_size},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        if not HAS_WHISPER:
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"asr-init","hypothesisId":"C","location":"modules/speech_to_text.py:get_speech_recognizer:whisper_not_available","message":"Whisper not available","data":{},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
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

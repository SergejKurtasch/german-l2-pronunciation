"""
Speech-to-Text module using macOS native Speech framework (SFSpeechRecognizer).
This is an alternative to Whisper that uses macOS built-in speech recognition.
"""

import sys
from pathlib import Path
from typing import Optional
import threading
import time

# Try to import Speech framework via PyObjC
# First check if we're on macOS
import sys
HAS_MACOS_SPEECH = False

if sys.platform == "darwin":  # macOS
    try:
        from Speech import (
            SFSpeechRecognizer,
            SFSpeechURLRecognitionRequest,
            SFSpeechRecognitionTask,
        )
        from Foundation import NSURL, NSLocale, NSRunLoop, CFRunLoopRun, CFRunLoopStop
        from AVFoundation import AVAudioFile, AVAudioFormat
        # Check for authorization constants
        try:
            from Speech import SFSpeechRecognizerAuthorizationStatus, SFSpeechRecognizerAuthorizationStatusAuthorized
            HAS_AUTHORIZATION_CONSTANTS = True
        except ImportError:
            HAS_AUTHORIZATION_CONSTANTS = False
        HAS_MACOS_SPEECH = True
    except ImportError as e:
        HAS_MACOS_SPEECH = False
        HAS_AUTHORIZATION_CONSTANTS = False
        # Only print warning if someone actually tries to use it
        # Don't print on import to avoid noise
else:
    # Not macOS, Speech framework not available
    HAS_MACOS_SPEECH = False
    HAS_AUTHORIZATION_CONSTANTS = False


class MacOSSpeechToTextRecognizer:
    """Speech-to-Text recognizer using macOS native Speech framework."""
    
    def __init__(self, language: str = "de-DE"):
        """
        Initialize macOS Speech recognizer.
        
        Args:
            language: Language code in BCP-47 format (e.g., 'de-DE' for German, 'en-US' for English)
        """
        if not HAS_MACOS_SPEECH:
            raise ImportError(
                "macOS Speech framework is not available. "
                "Install with: pip install pyobjc-framework-Speech"
            )
        
        self.language = language
        self.locale = NSLocale.alloc().initWithLocaleIdentifier_(language)
        self.recognizer = None
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize the speech recognizer."""
        try:
            # Check authorization status if available
            if HAS_AUTHORIZATION_CONSTANTS:
                pass
                # Note: Authorization is usually handled by macOS automatically on first use
                # We can't easily check it via PyObjC without more complex code
            
            self.recognizer = SFSpeechRecognizer.alloc().initWithLocale_(self.locale)
            
            if not self.recognizer:
                raise RuntimeError(f"Failed to create speech recognizer for locale {self.language}")
            
            is_available = self.recognizer.isAvailable()
            
            if not is_available:
                raise RuntimeError(
                    f"Speech recognition is not available for locale {self.language}. "
                    "Please check System Preferences > Keyboard > Dictation and ensure "
                    "the language is installed. Also check System Preferences > Security & Privacy > "
                    "Privacy > Speech Recognition and ensure your application has permission."
                )
            
            print(f"macOS Speech recognizer initialized for language: {self.language}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize macOS Speech recognizer: {e}")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: Optional[str] = None,  # Not used for macOS Speech, kept for compatibility
        verbose: bool = False
    ) -> str:
        """
        Transcribe audio file to text using macOS Speech framework.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'de-DE'). If None, uses the language from __init__
            task: Not used (kept for compatibility with Whisper interface)
            verbose: Whether to print progress information
            
        Returns:
            Transcribed text string
        """
        if self.recognizer is None:
            raise RuntimeError("Speech recognizer not initialized")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use provided language or default
        locale_to_use = self.locale
        if language:
            locale_to_use = NSLocale.alloc().initWithLocaleIdentifier_(language)
            # Create a new recognizer for this language if different
            if language != self.language:
                temp_recognizer = SFSpeechRecognizer.alloc().initWithLocale_(locale_to_use)
                if temp_recognizer and temp_recognizer.isAvailable():
                    recognizer_to_use = temp_recognizer
                else:
                    print(f"Warning: Language {language} not available, using {self.language}")
                    recognizer_to_use = self.recognizer
            else:
                recognizer_to_use = self.recognizer
        else:
            recognizer_to_use = self.recognizer
        
        # Convert audio path to NSURL
        audio_url = NSURL.fileURLWithPath_(audio_path)
        
        # Create recognition request
        request = SFSpeechURLRecognitionRequest.alloc().initWithURL_(audio_url)
        request.setShouldReportPartialResults_(False)  # We want final results only
        
        
        # Use threading and run loop to handle async recognition
        result_container = {'text': '', 'error': None, 'completed': False, 'lock': threading.Lock()}
        run_loop_ref = {'loop': None}
        
        def recognition_handler(result, error):
            """Handle recognition result."""
            
            with result_container['lock']:
                if error:
                    result_container['error'] = error
                    result_container['completed'] = True
                    if run_loop_ref['loop']:
                        CFRunLoopStop(run_loop_ref['loop'])
                elif result:
                    is_final = result.isFinal()
                    if is_final:
                        transcribed_text = result.bestTranscription().formattedString()
                        result_container['text'] = transcribed_text
                        result_container['completed'] = True
                        if run_loop_ref['loop']:
                            CFRunLoopStop(run_loop_ref['loop'])
                    else:
                        # Partial result - we'll wait for final
                        partial_text = result.bestTranscription().formattedString()
                        if verbose:
                            print(f"Partial transcription: {partial_text}")
                else:
                    pass
        
        # Perform recognition in a separate thread with its own run loop
        def recognition_thread():
            """Run recognition in a separate thread with run loop."""
            
            run_loop = NSRunLoop.currentRunLoop()
            run_loop_ref['loop'] = run_loop.getCFRunLoop()
            
            
            # Perform recognition
            task = recognizer_to_use.recognitionTaskWithRequest_resultHandler_(
                request,
                recognition_handler
            )
            
            
            if not task:
                with result_container['lock']:
                    result_container['error'] = RuntimeError("Failed to create recognition task")
                    result_container['completed'] = True
                    CFRunLoopStop(run_loop_ref['loop'])
                return
            
            
            # Store task reference to prevent garbage collection
            task_ref = {'task': task}
            
            # Run the run loop until completion
            # Use runMode:beforeDate: to allow the run loop to process events
            from Foundation import NSDate, NSDefaultRunLoopMode
            run_until = NSDate.dateWithTimeIntervalSinceNow_(3.0)  # 3 second timeout
            while True:
                
                with result_container['lock']:
                    if result_container['completed']:
                        break
                
                # Run loop for a short time
                run_until_short = NSDate.dateWithTimeIntervalSinceNow_(0.1)
                run_loop.runMode_beforeDate_(NSDefaultRunLoopMode, run_until_short)
                
                # Check if we've exceeded timeout
                if NSDate.date().timeIntervalSinceDate_(run_until) > 0:
                    break
            
        
        # Start recognition in a separate thread
        thread = threading.Thread(target=recognition_thread, daemon=False)
        thread.start()
        
        # Wait for completion (with timeout)
        timeout = 3  # 3 seconds timeout
        start_time = time.time()
        last_log_time = start_time
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            with result_container['lock']:
                if result_container['completed']:
                    break
                if elapsed > timeout:
                    if run_loop_ref['loop']:
                        CFRunLoopStop(run_loop_ref['loop'])
                    raise RuntimeError(f"Speech recognition timeout after {timeout} seconds")
            
            # Log progress every 5 seconds
            if current_time - last_log_time >= 5:
                last_log_time = current_time
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        # Wait for thread to finish
        thread.join(timeout=5)
        
        # Check for errors
        if result_container['error']:
            error_msg = str(result_container['error'])
            raise RuntimeError(f"Speech recognition failed: {error_msg}")
        
        transcribed_text = result_container['text'].strip()
        
        if verbose:
            print(f"Transcription completed: '{transcribed_text}'")
        
        return transcribed_text
    
    def transcribe_with_details(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: Optional[str] = None,
        verbose: bool = False
    ) -> dict:
        """
        Transcribe audio file and return full result with details.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'de-DE')
            task: Not used (kept for compatibility)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with:
            - text: Transcribed text
            - language: Language used
        """
        text = self.transcribe(audio_path, language, task, verbose)
        
        return {
            'text': text,
            'language': language or self.language
        }


# Global instance
_macos_speech_recognizer = None


def get_macos_speech_recognizer(
    language: Optional[str] = None
) -> Optional[MacOSSpeechToTextRecognizer]:
    """
    Get or create global macOS speech recognizer instance.
    
    Args:
        language: Language code (e.g., 'de-DE'). If None, uses config default.
        
    Returns:
        MacOSSpeechToTextRecognizer instance, or None if not available
    """
    global _macos_speech_recognizer
    
    if not HAS_MACOS_SPEECH:
        return None
    
    if _macos_speech_recognizer is None:
        # Try to get default from config
        if language is None:
            try:
                import config
                # Convert language code from config (e.g., 'de' -> 'de-DE')
                config_lang = getattr(config, 'ASR_LANGUAGE', 'de')
                if config_lang == 'de':
                    language = 'de-DE'
                elif config_lang == 'en':
                    language = 'en-US'
                else:
                    # Try to construct BCP-47 code
                    language = f"{config_lang}-{config_lang.upper()}"
            except ImportError:
                language = 'de-DE'
        
        if language is None:
            language = 'de-DE'
        
        try:
            _macos_speech_recognizer = MacOSSpeechToTextRecognizer(language=language)
        except Exception as e:
            print(f"Warning: Failed to initialize macOS speech recognizer: {e}")
            return None
    
    return _macos_speech_recognizer

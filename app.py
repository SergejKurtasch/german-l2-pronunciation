"""
German Pronunciation Diagnostic App (L2-Trainer)
Main application with Gradio interface.
"""

import gradio as gr
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import sys
import threading
import time
import json
import os
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules
import config
# VAD import commented out - VAD trimming is disabled
# from modules.vad_module import get_vad_detector
from modules.audio_normalizer import get_audio_normalizer
from modules.g2p_module import get_expected_phonemes, get_g2p_converter
from modules.phoneme_recognition import get_phoneme_recognizer
from modules.phoneme_filtering import get_phoneme_filter
from modules.forced_alignment import get_forced_aligner
from modules.alignment import needleman_wunsch_align
from modules.diagnostic_engine import get_diagnostic_engine
from modules.visualization import (
    create_side_by_side_comparison,
    create_colored_text,
    create_detailed_report,
    create_simple_phoneme_comparison,
    create_raw_phonemes_display,
    create_validation_comparison
)
from modules.phoneme_validator import get_optional_validator
from modules.speech_to_text import get_speech_recognizer
from modules.metrics import calculate_wer, calculate_per
from modules.mfa_alignment import get_mfa_aligner


def validate_single_phoneme(task_data):
    """
    Wrapper function for parallel phoneme validation.
    
    Args:
        task_data: Dictionary containing validation task parameters:
            - audio_segment: Audio segment to validate
            - phoneme_pair: Phoneme pair name
            - expected_phoneme: Expected phoneme
            - suspected_phoneme: Suspected phoneme
            - index: Index in aligned_pairs
            - segment_index: Index in recognized_segments
    
    Returns:
        Dictionary with validation result and metadata:
            - index: Original index in aligned_pairs
            - segment_index: Index in recognized_segments
            - validation_result: Result from validate_phoneme_segment
    """
    try:
        validator = task_data['validator']
        result = validator.validate_phoneme_segment(
            task_data['audio_segment'],
            phoneme_pair=task_data['phoneme_pair'],
            expected_phoneme=task_data['expected_phoneme'],
            suspected_phoneme=task_data['suspected_phoneme'],
            sr=task_data['sr']
        )
        return {
            'index': task_data['index'],
            'segment_index': task_data['segment_index'],
            'validation_result': result,
            'expected_phoneme': task_data['expected_phoneme'],
            'recognized_phoneme': task_data['suspected_phoneme'],
            'phoneme_pair': task_data['phoneme_pair'],
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'index': task_data['index'],
            'segment_index': task_data.get('segment_index', -1),
            'validation_result': {
                'is_correct': None,
                'confidence': 0.0,
                'error': str(e)
            },
            'expected_phoneme': task_data.get('expected_phoneme', ''),
            'recognized_phoneme': task_data.get('suspected_phoneme', ''),
            'phoneme_pair': task_data.get('phoneme_pair', ''),
            'error': str(e)
        }


# Global instances
# vad_detector = None  # VAD disabled
audio_normalizer = None
phoneme_recognizer = None
phoneme_filter = None
mfa_aligner = None
forced_aligner = None
diagnostic_engine = None
optional_validator = None
asr_recognizer = None


def initialize_asr_only():
    """Initialize only ASR (Whisper or macOS Speech) for fast startup."""
    global asr_recognizer    
    if asr_recognizer is None and config.ASR_ENABLED:
        try:
            requested_engine = getattr(config, 'ASR_ENGINE', 'whisper')            
            asr_recognizer = get_speech_recognizer(
                model_size=getattr(config, 'ASR_MODEL', 'medium'),
                device=getattr(config, 'ASR_DEVICE', None),
                engine=requested_engine
            )            
            if asr_recognizer:
                # Determine which engine was actually used (may differ from requested)
                actual_engine = requested_engine
                # Check if it's macOS wrapper by checking if it has recognizer attribute
                if hasattr(asr_recognizer, 'recognizer'):
                    actual_engine = "macos"
                else:
                    actual_engine = "whisper"                
                if actual_engine == "macos":
                    print(f"ASR recognizer (macOS Speech) initialized")
                else:
                    if requested_engine == "macos" and actual_engine == "whisper":
                        print(f"ASR recognizer (Whisper {getattr(config, 'ASR_MODEL', 'medium')}) initialized (macOS Speech not available, using fallback)")
                    else:
                        print(f"ASR recognizer (Whisper {getattr(config, 'ASR_MODEL', 'medium')}) initialized")
            else:
                print(f"Warning: ASR recognizer not available (neither {requested_engine} nor Whisper available)")
        except Exception as e:
            print(f"Warning: ASR recognizer initialization failed: {e}")
            asr_recognizer = None


def load_dictionaries_in_background():
    """Load G2P dictionaries in background after ASR is loaded."""
    from modules.g2p_module import load_g2p_dictionaries
    
    print("Starting background dictionary loading...")
    try:
        load_g2p_dictionaries()
        print("All dictionaries loaded successfully in background!")
    except Exception as e:
        print(f"Warning: Dictionary loading failed: {e}")


def load_phoneme_model_in_background():
    """Load Wav2Vec2 phoneme recognition model in background."""
    global phoneme_recognizer
    
    print("Stage 3: Loading phoneme recognition model (Wav2Vec2)...")    
    try:
        if phoneme_recognizer is None:
            model_load_start = time.time()
            phoneme_recognizer = get_phoneme_recognizer(
                model_name=config.MODEL_NAME,
                device=config.MODEL_DEVICE if config.MODEL_DEVICE != "auto" else None
            )
            model_load_elapsed = (time.time() - model_load_start) * 1000
            print(f"Phoneme recognition model loaded successfully! (took {model_load_elapsed/1000:.2f}s)")
        else:
            print("Phoneme recognition model already loaded.")
    except Exception as e:
        print(f"Warning: Phoneme model loading failed: {e}")


def load_mfa_in_background():
    """Load MFA aligner in background."""
    global mfa_aligner
    
    print("Stage 4: Loading MFA aligner...")
    try:
        if mfa_aligner is None:
            # Check if MFA is available in conda environment
            import subprocess
            import shutil
            
            conda_env = config.MFA_CONDA_ENV
            mfa_available = False
            
            # Find conda executable
            conda_cmd = shutil.which("conda")
            if not conda_cmd:
                # Try common conda locations
                possible_paths = [
                    Path.home() / "miniforge3" / "bin" / "conda",
                    Path.home() / "miniforge3" / "condabin" / "conda",
                    Path.home() / "miniforge" / "bin" / "conda",
                    Path.home() / "anaconda3" / "bin" / "conda",
                    Path.home() / "miniconda3" / "bin" / "conda",
                    Path("/opt/homebrew/Caskroom/miniforge/base/bin/conda"),
                    Path("/usr/local/Caskroom/miniforge/base/bin/conda"),
                ]
                for path in possible_paths:
                    if path.exists():
                        conda_cmd = str(path)
                        break
                
                # Try CONDA_EXE environment variable
                import os
                conda_exe = os.environ.get("CONDA_EXE")
                if conda_exe and Path(conda_exe).exists():
                    conda_cmd = conda_exe
            
            # Try to find MFA binary directly in common conda locations
            possible_mfa_paths = [
                Path.home() / "miniforge3" / "envs" / conda_env / "bin" / "mfa",
                Path.home() / "miniforge" / "envs" / conda_env / "bin" / "mfa",
                Path.home() / "anaconda3" / "envs" / conda_env / "bin" / "mfa",
                Path.home() / "miniconda3" / "envs" / conda_env / "bin" / "mfa",
                Path("/opt/homebrew/Caskroom/miniforge/base/envs") / conda_env / "bin" / "mfa",
                Path("/usr/local/Caskroom/miniforge/base/envs") / conda_env / "bin" / "mfa",
            ]
            
            for mfa_path in possible_mfa_paths:
                if mfa_path.exists():
                    mfa_available = True
                    break
            
            if not mfa_available and shutil.which("mfa"):
                mfa_available = True
            elif not mfa_available and conda_cmd:
                # Try using conda to check
                try:
                    result = subprocess.run(
                        [conda_cmd, "run", "-n", conda_env, "which", "mfa"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        mfa_path = result.stdout.strip()
                        if mfa_path and Path(mfa_path).exists():
                            mfa_available = True
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            
            if not mfa_available:
                print(f"Warning: MFA not found in conda environment '{conda_env}'.")
                if not conda_cmd:
                    print("Warning: conda not found. Please ensure conda is installed and accessible.")
                    print("  Common locations: ~/miniforge3/bin/conda, ~/anaconda3/bin/conda")
                    print("  Or set CONDA_EXE environment variable")
                else:
                    print(f"  MFA should be installed in environment '{conda_env}'")
                    print(f"  To check: conda run -n {conda_env} which mfa")
                    print(f"  To install: conda install -c conda-forge montreal-forced-aligner -n {conda_env} -y")
            
            # Initialize MFA aligner (will handle missing binary gracefully)
            mfa_aligner = get_mfa_aligner()
            if mfa_aligner:
                print("MFA aligner loaded successfully!")
            else:
                print("Warning: MFA aligner initialization failed")
        else:
            print("MFA aligner already loaded.")
    except Exception as e:
        print(f"Warning: MFA aligner loading failed: {e}")
        import traceback
        traceback.print_exc()
        mfa_aligner = None


def collapse_consecutive_duplicates(phonemes: List[str]) -> List[str]:
    """
    Collapse consecutive duplicate phonemes (same logic as CTC collapse).
    This ensures that expected and recognized phonemes are processed consistently.
    
    Args:
        phonemes: List of phoneme strings
        
    Returns:
        List of phonemes with consecutive duplicates collapsed
    """
    if not phonemes:
        return phonemes
    
    collapsed = []
    prev_phoneme = None
    
    for phoneme in phonemes:
        # Skip empty phonemes
        if not phoneme or not phoneme.strip():
            continue
        
        # If different from previous, add it
        if phoneme != prev_phoneme:
            collapsed.append(phoneme)
            prev_phoneme = phoneme
        # If same as previous, skip it (CTC collapse)
        # prev_phoneme stays the same to allow same token later
    
    return collapsed


def initialize_components():
    """Initialize global components."""
    # global vad_detector, audio_normalizer, phoneme_recognizer, phoneme_filter, forced_aligner, diagnostic_engine, optional_validator, asr_recognizer
    global audio_normalizer, phoneme_recognizer, phoneme_filter, forced_aligner, diagnostic_engine, optional_validator, asr_recognizer, mfa_aligner
    
    import json, time
    init_components_start = time.time()
    
    if audio_normalizer is None:
        try:
            comp_start = time.time()
            audio_normalizer = get_audio_normalizer()
            comp_elapsed = (time.time() - comp_start) * 1000
            print("Audio normalizer initialized")
        except Exception as e:
            print(f"Warning: Audio normalizer initialization failed: {e}")
            audio_normalizer = None
    
    # VAD initialization commented out - VAD trimming is disabled
    # if vad_detector is None:
    #     try:
    #         vad_detector = get_vad_detector(method=config.VAD_METHOD)
    #         print("VAD detector initialized")
    #     except Exception as e:
    #         print(f"Warning: VAD initialization failed: {e}")
    #         vad_detector = None
    
    if phoneme_recognizer is None:
        try:
            comp_start = time.time()
            phoneme_recognizer = get_phoneme_recognizer(
                model_name=config.MODEL_NAME,
                device=config.MODEL_DEVICE if config.MODEL_DEVICE != "auto" else None
            )
            comp_elapsed = (time.time() - comp_start) * 1000
            print(f"Phoneme recognizer (Wav2Vec2 XLSR-53 eSpeak) initialized with model: {phoneme_recognizer.model_name}")
        except Exception as e:
            print(f"Error: Phoneme recognizer initialization failed: {e}")
            raise
    
    if phoneme_filter is None:
        comp_start = time.time()
        phoneme_filter = get_phoneme_filter(
            whitelist=config.PHONEME_WHITELIST,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        comp_elapsed = (time.time() - comp_start) * 1000
        print("Phoneme filter initialized")
    
    if forced_aligner is None:
        comp_start = time.time()
        forced_aligner = get_forced_aligner(blank_id=config.FORCED_ALIGNMENT_BLANK_ID)
        comp_elapsed = (time.time() - comp_start) * 1000
        print("Forced aligner initialized")
    
    if diagnostic_engine is None:
        comp_start = time.time()
        diagnostic_engine = get_diagnostic_engine()
        comp_elapsed = (time.time() - comp_start) * 1000
        print("Diagnostic engine initialized")
    
    if optional_validator is None:
        comp_start = time.time()
        optional_validator = get_optional_validator()
        comp_elapsed = (time.time() - comp_start) * 1000
        print("Optional validator initialized")
    
    # Preload G2P dictionaries to avoid lazy loading delay
    comp_start = time.time()
    from modules.g2p_module import get_g2p_converter
    g2p_converter = get_g2p_converter(load_dicts_immediately=False)
    if not g2p_converter._dicts_loaded:
        print("Preloading G2P dictionaries...")
        g2p_converter._load_dictionaries()
        print("G2P dictionaries preloaded!")
    comp_elapsed = (time.time() - comp_start) * 1000    
    # ASR is loaded separately in initialize_asr_only()
    # Initialize ASR here only if not already loaded
    if asr_recognizer is None and config.ASR_ENABLED:
        initialize_asr_only()
    
    # Initialize MFA aligner if enabled and not already loaded
    if mfa_aligner is None and config.MFA_ENABLED:
        try:
            mfa_aligner = get_mfa_aligner()
            if mfa_aligner:
                print("MFA aligner initialized")
        except Exception as e:
            print(f"Warning: MFA aligner initialization failed: {e}")
            mfa_aligner = None


def normalize_chat_history(chat_history: Optional[List]) -> List:
    """
    Normalize chat history to the format expected by Gradio Chatbot.
    Converts old tuple format [user_msg, assistant_msg] to dict format.
    
    Args:
        chat_history: Chat history in any format (None, list of tuples, or list of dicts)
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    if chat_history is None:
        return []
    
    normalized = []
    for msg in chat_history:
        if isinstance(msg, dict):
            # Already in correct format
            normalized.append(msg)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Old tuple format [user_msg, assistant_msg] - convert to dict format
            user_msg, assistant_msg = msg
            normalized.append({"role": "user", "content": user_msg})
            normalized.append({"role": "assistant", "content": assistant_msg})
        else:
            # Unknown format - skip
            continue
    
    return normalized


def add_to_chat_history(chat_history: Optional[List], user_message: str, assistant_message: str) -> List:
    """
    Add a new message pair to chat history in the correct format.
    
    Args:
        chat_history: Current chat history (any format)
        user_message: User message text
        assistant_message: Assistant message text (HTML)
        
    Returns:
        Updated chat history in normalized format
    """
    normalized = normalize_chat_history(chat_history)
    normalized.append({"role": "user", "content": user_message})
    normalized.append({"role": "assistant", "content": assistant_message})
    return normalized


def process_pronunciation(
    text: str,
    audio_file: Optional[Tuple[int, np.ndarray]] = None,
    enable_validation: bool = False,
    chat_history: Optional[List] = None
) -> Tuple[List, str, str]:
    """
    Process pronunciation validation.
    
    Args:
        text: German text input
        audio_file: Tuple of (sample_rate, audio_array) from Gradio
        enable_validation: Whether to enable optional validation through trained models
        chat_history: Current chat history
        
    Returns:
        Tuple of:
        1. Updated chat history (list of [user_message, assistant_message] tuples)
        2. Original text input (to preserve it)
        3. Original audio input (to preserve it)
    """
    start_time = time.time()
    
    # Initialize output variables to avoid NameError
    side_by_side_html = ""
    colored_text_html = ""
    detailed_report_html = ""
    technical_html = ""
    raw_phonemes_html = ""
    
    # Check if text is empty - if so, we'll use ASR to get text from audio
    text_is_empty = not text or not text.strip()
    
    if audio_file is None:
        error_html = "<div style='color: orange; padding: 10px;'>Please record or upload audio.</div>"
        user_message = f"Text: {text if text else 'No text'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
        chat_history = add_to_chat_history(chat_history, user_message, error_html)
        return (chat_history, text, audio_file)
    
    try:
        # Initialize components
        init_start = time.time()
        initialize_components()
        init_elapsed = (time.time() - init_start) * 1000        
        # Extract audio
        if isinstance(audio_file, tuple):
            sample_rate, audio_array = audio_file
        else:
            error_html = "<div style='color: red; padding: 10px;'>Invalid audio format.</div>"
            user_message = f"Text: {text if text else 'No text'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
            chat_history = add_to_chat_history(chat_history, user_message, error_html)
            return (chat_history, text, audio_file)
        
        # Save audio to temporary file
        audio_save_start = time.time()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, sample_rate)
        audio_save_elapsed = (time.time() - audio_save_start) * 1000        
        try:
            # Stage 0: Audio normalization (for AGC issues) - COMMENTED OUT
            # normalized_audio_path = tmp_path
            # if audio_normalizer is not None and config.ENABLE_AUDIO_NORMALIZATION:
            #     try:
            #         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as normalized_file:
            #             normalized_path = normalized_file.name
            #         normalized_audio_path = audio_normalizer.process_audio_file(
            #             tmp_path,
            #             normalized_path,
            #             sample_rate,
            #             compress_peaks=config.NORMALIZE_COMPRESS_PEAKS,
            #             peak_compression_ratio=config.NORMALIZE_PEAK_COMPRESSION_RATIO,
            #             peak_compression_duration_ms=config.NORMALIZE_PEAK_COMPRESSION_DURATION_MS,
            #             normalize_method=config.NORMALIZE_METHOD
            #         )
            #         print(f"Audio normalization: Compressed peaks and normalized")
            #     except Exception as e:
            #         print(f"Warning: Audio normalization failed: {e}")
            #         normalized_audio_path = tmp_path
            normalized_audio_path = tmp_path  # Use original audio without normalization
            
            # Stage 1: VAD - Trim noise (DISABLED - commented out)
            # VAD trimming is disabled - using normalized audio (or original) directly
            vad_info = {'enabled': False, 'reason': 'VAD disabled'}
            trimmed_audio_path = normalized_audio_path  # Use normalized audio directly without VAD trimming
            # vad_info = {}
            # trimmed_audio_path = normalized_audio_path
            # if vad_detector is not None:
            #     try:
            #         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as trimmed_file:
            #             trimmed_path = trimmed_file.name
            #         # Use ultra-conservative padding
            #         trimmed_audio_path = vad_detector.trim_audio(
            #             normalized_audio_path,  # Use normalized audio for VAD
            #             trimmed_path,
            #             sample_rate,
            #             padding_ms=config.VAD_PADDING_MS  # Will use VAD_PADDING_END_MS for end internally
            #         )
            #         vad_info = {'enabled': True, 'trimmed_path': trimmed_audio_path}
            #         print(f"VAD: Audio trimmed")
            #     except Exception as e:
            #         print(f"Warning: VAD failed: {e}")
            #         trimmed_audio_path = tmp_path
            #         vad_info = {'enabled': False, 'error': str(e)}
            # else:
            #     vad_info = {'enabled': False, 'reason': 'VAD not available'}
            
            # Stage 2: ASR - Speech-to-Text recognition
            asr_start = time.time()
            recognized_text = None
            wer_result = None
            
            # If text is empty, we MUST use ASR to get text from audio
            if text_is_empty:
                if asr_recognizer and config.ASR_ENABLED:
                    try:
                        recognized_text = asr_recognizer.transcribe(
                            trimmed_audio_path,
                            language=config.ASR_LANGUAGE
                        )
                        asr_elapsed = (time.time() - asr_start) * 1000
                        # Determine ASR engine with detailed logging
                        asr_engine = "unknown"
                        asr_device = "unknown"
                        has_recognizer = hasattr(asr_recognizer, 'recognizer')
                        has_model = hasattr(asr_recognizer, 'model')
                        asr_type = str(type(asr_recognizer))
                        
                        if has_recognizer:
                            asr_engine = "macos"
                            asr_device = "macos_native"  # macOS Speech uses native framework
                        elif has_model:
                            asr_engine = "whisper"
                            # Get device from Whisper recognizer
                            if hasattr(asr_recognizer, 'device'):
                                asr_device = asr_recognizer.device
                            else:
                                asr_device = "unknown"
                        print(f"ASR: Recognized text (from audio): '{recognized_text}'")
                        
                        if not recognized_text or not recognized_text.strip():
                            error_html = "<div style='color: orange; padding: 10px;'>Could not recognize text from audio. Please try again or enter text manually.</div>"
                            user_message = f"Text: {text if text else 'Audio input'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                            chat_history = add_to_chat_history(chat_history, user_message, error_html)
                            return (chat_history, text, audio_file)
                    except Exception as e:
                        print(f"Error: ASR failed: {e}")
                        error_html = f"<div style='color: red; padding: 10px;'>Failed to recognize text from audio: {str(e)}</div>"
                        user_message = f"Text: {text if text else 'Audio input'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                        chat_history = add_to_chat_history(chat_history, user_message, error_html)
                        return (chat_history, text, audio_file)
                else:
                    error_html = "<div style='color: orange; padding: 10px;'>ASR is not available. Please enter text manually or enable ASR in configuration.</div>"
                    user_message = f"Text: {text if text else 'Audio input'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                    chat_history = add_to_chat_history(chat_history, user_message, error_html)
                    return (chat_history, text, audio_file)
            elif asr_recognizer and config.ASR_ENABLED and config.ASR_ALWAYS_RUN:
                # Text is provided, ASR is optional (for comparison)
                # Only run if ASR_ALWAYS_RUN is enabled to save time
                try:
                    recognized_text = asr_recognizer.transcribe(
                        trimmed_audio_path,
                        language=config.ASR_LANGUAGE
                    )
                    asr_elapsed = (time.time() - asr_start) * 1000
                    # Determine ASR engine with detailed logging
                    asr_engine = "unknown"
                    asr_device = "unknown"
                    has_recognizer = hasattr(asr_recognizer, 'recognizer')
                    has_model = hasattr(asr_recognizer, 'model')
                    asr_type = str(type(asr_recognizer))
                    
                    if has_recognizer:
                        asr_engine = "macos"
                        asr_device = "macos_native"  # macOS Speech uses native framework
                    elif has_model:
                        asr_engine = "whisper"
                        # Get device from Whisper recognizer
                        if hasattr(asr_recognizer, 'device'):
                            asr_device = asr_recognizer.device
                        else:
                            asr_device = "unknown"
                    print(f"ASR: Recognized text: '{recognized_text}'")
                    
                    # Stage 3: WER Calculation (only if text was provided)
                    if recognized_text:
                        wer_start = time.time()
                        try:
                            wer_result = calculate_wer(text, recognized_text)
                            wer_elapsed = (time.time() - wer_start) * 1000
                            print(f"WER: {wer_result['wer']:.2%} (Substitutions: {wer_result['substitutions']}, "
                                  f"Deletions: {wer_result['deletions']}, Insertions: {wer_result['insertions']})")
                        except Exception as e:
                            wer_elapsed = (time.time() - wer_start) * 1000
                            print(f"Error: WER calculation failed: {e}")
                            wer_result = None
                except Exception as e:
                    print(f"Warning: ASR failed: {e}")
                    recognized_text = None
                    wer_result = None
            else:
                # Skip ASR if text is provided and ASR_ALWAYS_RUN is False
                recognized_text = None
                wer_result = None
            
            # Stage 4: Check WER threshold - skip phoneme analysis if WER is too high
            # Skip this check if text was empty (WER not calculated)
            if not text_is_empty and wer_result and wer_result['wer'] > config.WER_THRESHOLD and config.WER_SKIP_PHONEME_ANALYSIS:
                # High WER - show only text comparison
                from modules.visualization import create_text_comparison_view
                
                # Create simplified view
                try:
                    comparison_html = create_text_comparison_view(text, recognized_text or "", wer_result)
                except Exception as e:
                    print(f"Error: Failed to create text comparison view: {e}")
                    comparison_html = f"<div style='color: red; padding: 10px;'>Error creating text comparison: {str(e)}</div>"
                empty_html = "<div style='color: gray; padding: 10px;'>Phoneme analysis skipped due to high WER.</div>"
                
                technical_html = f"""
                <div style='padding: 10px; background: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>
                    <h4>High Word Error Rate Detected</h4>
                    <p><strong>WER:</strong> {wer_result['wer']:.2%} (Threshold: {config.WER_THRESHOLD:.2%})</p>
                    <p>Phoneme analysis has been skipped because the recognized text differs significantly from the expected text.</p>
                    <ul>
                        <li><strong>Expected:</strong> {text}</li>
                        <li><strong>Recognized:</strong> {recognized_text or 'N/A'}</li>
                        <li><strong>Substitutions:</strong> {wer_result['substitutions']}</li>
                        <li><strong>Deletions:</strong> {wer_result['deletions']}</li>
                        <li><strong>Insertions:</strong> {wer_result['insertions']}</li>
                        <li><strong>Correct words:</strong> {wer_result['hits']} / {wer_result['total_reference_words']}</li>
                    </ul>
                </div>
                """
                
                # Create empty raw phonemes display for high WER case
                raw_phonemes_html = "<div style='color: gray; padding: 10px;'>Raw phonemes not available (phoneme analysis skipped due to high WER).</div>"
                
                # Build chat message for high WER case
                assistant_message = f"""
                <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
                    {comparison_html}
                    {technical_html}
                </div>
                """
                
                user_message = f"Text: {text}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                chat_history = add_to_chat_history(chat_history, user_message, assistant_message)
                return (chat_history, text, audio_file)
            
            # Stage 5: G2P - Get phonemes from recognized text (or expected text if ASR not available)
            # Use recognized text for phoneme analysis if available, otherwise use expected text
            g2p_start = time.time()
            text_for_phonemes = recognized_text if recognized_text else text
            expected_phonemes_dict = get_expected_phonemes(text_for_phonemes)
            expected_phonemes = [ph.get('phoneme', '') for ph in expected_phonemes_dict]
            # Apply CTC collapse logic to expected phonemes (same as model does)
            expected_phonemes = collapse_consecutive_duplicates(expected_phonemes)
            g2p_elapsed = (time.time() - g2p_start) * 1000
            print(f"Expected phonemes (from {'recognized' if recognized_text else 'expected'} text): {len(expected_phonemes)}")
            
            if not expected_phonemes:
                error_html = "<div style='color: red; padding: 10px;'>Failed to extract expected phonemes from text.</div>"
                user_message = f"Text: {text_for_phonemes}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                chat_history = add_to_chat_history(chat_history, user_message, error_html)
                return (chat_history, text, audio_file)
            
            # Stage 3: Phoneme Recognition (Wav2Vec2 XLSR-53 eSpeak)
            phoneme_rec_start = time.time()
            logits, emissions = phoneme_recognizer.recognize_phonemes(
                trimmed_audio_path,
                sample_rate=config.SAMPLE_RATE
            )
            vocab = phoneme_recognizer.get_vocab()
            
            # Decode phonemes (for display)
            decode_start = time.time()
            decoded_phonemes_str = phoneme_recognizer.decode_phonemes(logits)
            decode_elapsed = (time.time() - decode_start) * 1000
            phoneme_rec_elapsed = (time.time() - phoneme_rec_start) * 1000            
            raw_phonemes = decoded_phonemes_str.split()
            # Apply CTC collapse logic to recognized phonemes (for consistency, though model already does this)
            raw_phonemes = collapse_consecutive_duplicates(raw_phonemes)            
            print(f"Raw phonemes: {len(raw_phonemes)}")
            
            # Stage 4: Multi-level Filtering
            # Note: Filtering is kept for forced alignment (uses ARPABET), but raw_phonemes are used for display and alignment
            filter_start = time.time()
            filtered_phonemes = phoneme_filter.filter_combined(
                logits,
                raw_phonemes,
                vocab
            )
            
            recognized_phonemes = [ph.get('phoneme', '') for ph in filtered_phonemes]
            filter_elapsed = (time.time() - filter_start) * 1000
            print(f"Filtered phonemes: {len(recognized_phonemes)}")
            
            # Use raw_phonemes for validation - model already outputs accurate IPA phonemes
            if not raw_phonemes:
                error_html = "<div style='color: orange; padding: 10px;'>No phonemes recognized. Audio may be unclear.</div>"
                # Create raw phonemes display even if filtered is empty
                raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
                assistant_message = f"""
                <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
                    {error_html}
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; color: #0066cc; text-decoration: underline;">Show phonemes</summary>
                        <div style="margin-top: 10px;">
                            {raw_phonemes_html}
                        </div>
                    </details>
                </div>
                """
                user_message = f"Text: {text_for_phonemes}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
                chat_history = add_to_chat_history(chat_history, user_message, assistant_message)
                return (chat_history, text, audio_file)
            
            # Stage 5: Forced Alignment (for recognized phonemes)
            # Load waveform for forced alignment
            alignment_start = time.time()
            waveform_load_start = time.time()
            waveform, sr = librosa.load(trimmed_audio_path, sr=config.SAMPLE_RATE, mono=True)
            waveform_load_elapsed = (time.time() - waveform_load_start) * 1000
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            
            recognized_segments = []
            alignment_method = "CTC"
            
            # Use MFA alignment only when:
            # 1. MFA_ENABLED = True in config (controls both loading and usage)
            # 2. Validation is enabled (MFA provides better accuracy for validation)
            # 3. MFA aligner is available and text is provided
            use_mfa = (config.MFA_ENABLED and 
                      enable_validation and 
                      mfa_aligner is not None and 
                      text and text.strip())
            
            # Choose alignment method: MFA or CTC
            if use_mfa:
                # Use MFA alignment with original text from interface
                try:
                    mfa_align_start = time.time()
                    # Get expected phonemes for MFA (from original text)
                    expected_phonemes_for_mfa = [ph.get('phoneme', '') for ph in get_expected_phonemes(text)]
                    recognized_segments = mfa_aligner.extract_phoneme_segments(
                        Path(trimmed_audio_path),
                        text.strip(),
                        expected_phonemes_for_mfa,
                        config.SAMPLE_RATE
                    )
                    mfa_align_elapsed = (time.time() - mfa_align_start) * 1000
                    alignment_method = "MFA"
                    
                    # Log MFA alignment latency
                    audio_duration = len(waveform) / config.SAMPLE_RATE
                    log_path = PROJECT_ROOT / ".cursor" / "debug.log"
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({
                            "sessionId": "debug-session",
                            "runId": "performance",
                            "hypothesisId": "ALIGNMENT_LATENCY",
                            "location": "app.py:process_pronunciation:alignment",
                            "message": "Alignment completed",
                            "data": {
                                "method": "MFA",
                                "latency_ms": mfa_align_elapsed,
                                "segments_count": len(recognized_segments),
                                "audio_duration_seconds": audio_duration
                            },
                            "timestamp": int(time.time() * 1000),
                            "elapsed_ms": int(mfa_align_elapsed)
                        }) + '\n')
                    
                    print(f"MFA alignment completed: {len(recognized_segments)} segments in {mfa_align_elapsed:.0f}ms")
                except Exception as e:
                    print(f"Warning: MFA alignment failed: {e}, falling back to CTC")
                    import traceback
                    traceback.print_exc()
                    use_mfa = False  # Fallback to CTC
            
            # Use CTC alignment (default or fallback from MFA)
            if not use_mfa or not recognized_segments:
                # Extract segments for recognized phonemes using CTC
                # Use ARPABET phonemes for forced alignment (vocab contains ARPABET tokens)
                recognized_phonemes_arpabet = [
                    ph.get('phoneme_arpabet', ph.get('phoneme', '')) 
                    for ph in filtered_phonemes 
                    if ph.get('phoneme_arpabet') or ph.get('phoneme')
                ]
                if len(recognized_phonemes_arpabet) > 0:
                    try:
                        ctc_align_start = time.time()
                        recognized_segments = forced_aligner.extract_phoneme_segments(
                            waveform_tensor,
                            recognized_phonemes_arpabet,
                            emissions,
                            vocab,
                            config.SAMPLE_RATE
                        )
                        ctc_align_elapsed = (time.time() - ctc_align_start) * 1000
                        alignment_method = "CTC"
                        
                        # Log CTC alignment latency
                        audio_duration = len(waveform) / config.SAMPLE_RATE
                        log_path = PROJECT_ROOT / ".cursor" / "debug.log"
                        with open(log_path, 'a') as f:
                            f.write(json.dumps({
                                "sessionId": "debug-session",
                                "runId": "performance",
                                "hypothesisId": "ALIGNMENT_LATENCY",
                                "location": "app.py:process_pronunciation:alignment",
                                "message": "Alignment completed",
                                "data": {
                                    "method": "CTC",
                                    "latency_ms": ctc_align_elapsed,
                                    "segments_count": len(recognized_segments),
                                    "audio_duration_seconds": audio_duration
                                },
                                "timestamp": int(time.time() * 1000),
                                "elapsed_ms": int(ctc_align_elapsed)
                            }) + '\n')
                        
                        # Update segment labels to IPA - use raw_phonemes for accurate IPA labels
                        # Note: segment count may differ from raw_phonemes count due to forced alignment
                        for i, segment in enumerate(recognized_segments):
                            if i < len(raw_phonemes):
                                segment.label = raw_phonemes[i]
                    except Exception as e:
                        print(f"Warning: CTC forced alignment failed: {e}")
            
            alignment_elapsed = (time.time() - alignment_start) * 1000            
            # Stage 6: Needleman-Wunsch Alignment
            # Use raw_phonemes directly - model already outputs accurate IPA phonemes
            # Now using phoneme similarity matrix for more accurate alignment
            nw_start = time.time()
            aligned_pairs, alignment_score = needleman_wunsch_align(
                expected_phonemes,
                raw_phonemes,
                match_score=config.NW_MATCH_SCORE,
                mismatch_score=config.NW_MISMATCH_SCORE,
                gap_penalty=config.NW_GAP_PENALTY,
                use_similarity_matrix=config.USE_PHONEME_SIMILARITY
            )
            nw_elapsed = (time.time() - nw_start) * 1000            
            print(f"Aligned pairs: {len(aligned_pairs)}, score: {alignment_score:.2f}")
            
            # Stage 7: PER Calculation
            per_start = time.time()
            per_result = calculate_per(aligned_pairs)
            per_elapsed = (time.time() - per_start) * 1000
            print(f"PER: {per_result['per']:.2%} (Substitutions: {per_result['substitutions']}, "
                  f"Deletions: {per_result['deletions']}, Insertions: {per_result['insertions']})")
            
            # Stage 8: Diagnostic Analysis
            diagnostic_start = time.time()
            diagnostic_results = diagnostic_engine.analyze_pronunciation(aligned_pairs)
            diagnostic_elapsed = (time.time() - diagnostic_start) * 1000            
            # Store aligned_pairs before validation for comparison
            aligned_pairs_before_validation = [(exp, rec) for exp, rec in aligned_pairs] if enable_validation else None
            diagnostic_results_before_validation = [dict(dr) for dr in diagnostic_results] if enable_validation else None
            
            # Stage 9: Optional Validation (Parallelized)
            validation_start = time.time()
            validation_count = 0
            validation_corrected_count = 0
            if enable_validation and optional_validator:
                # For each mismatch in aligned_pairs, try to validate with trained model
                print(f"Optional validation enabled - checking {len(aligned_pairs)} aligned pairs")
                
                # Step 1: Collect all validation tasks
                validation_tasks = []
                segment_index = 0
                MIN_SEGMENT_LENGTH = 100  # samples
                CONTEXT_MS = 100.0  # Use 100ms context window
                
                for i, (expected_ph, recognized_ph) in enumerate(aligned_pairs):
                    # Skip if match, missing (None), or word boundary
                    if expected_ph == recognized_ph or expected_ph is None or recognized_ph is None:
                        # Advance segment index if recognized phoneme exists
                        if recognized_ph is not None and recognized_ph != '||':
                            segment_index += 1
                        continue
                    
                    # Skip word boundaries
                    if expected_ph == '||' or recognized_ph == '||':
                        continue
                    
                    # Check if trained model exists for this phoneme pair
                    if optional_validator.has_trained_model(expected_ph, recognized_ph):
                        # Get proper phoneme pair name
                        phoneme_pair = optional_validator.get_phoneme_pair(expected_ph, recognized_ph)
                        if phoneme_pair is None:
                            print(f"Warning: Could not get phoneme pair for {expected_ph} -> {recognized_ph}")
                            segment_index += 1
                            continue
                        
                        # Find corresponding segment using index
                        segment = None
                        if segment_index < len(recognized_segments):
                            segment = recognized_segments[segment_index]
                        
                        if segment:
                            # Extract audio segment
                            start_sample = int(segment.start_time * config.SAMPLE_RATE)
                            end_sample = int(segment.end_time * config.SAMPLE_RATE)
                            audio_segment = waveform[start_sample:end_sample].copy()
                            
                            # Fallback for empty or very short segments (< 100 samples = ~6ms)
                            # This happens when forced aligner fails to determine boundaries (e.g., at end of audio)
                            if len(audio_segment) < MIN_SEGMENT_LENGTH:
                                # Use segment start_time as center point, or estimate from index
                                center_time = segment.start_time if segment.start_time > 0 else (segment_index / len(recognized_segments)) * (len(waveform) / config.SAMPLE_RATE)
                                
                                # Extract context window around the position
                                context_samples = int(CONTEXT_MS / 1000 * config.SAMPLE_RATE)
                                half_context = context_samples // 2
                                center_sample = int(center_time * config.SAMPLE_RATE)
                                
                                fallback_start = max(0, center_sample - half_context)
                                fallback_end = min(len(waveform), center_sample + half_context)
                                audio_segment = waveform[fallback_start:fallback_end].copy()
                            
                            # Add task to validation queue
                            validation_tasks.append({
                                'index': i,
                                'audio_segment': audio_segment,
                                'phoneme_pair': phoneme_pair,
                                'expected_phoneme': expected_ph,
                                'suspected_phoneme': recognized_ph,
                                'segment_index': segment_index,
                                'validator': optional_validator,
                                'sr': config.SAMPLE_RATE
                            })
                        else:
                            print(f"Warning: No segment found for {recognized_ph} at index {segment_index}")
                        
                        # Advance segment index
                        segment_index += 1
                    else:
                        # No model for this pair, advance segment index
                        if recognized_ph is not None and recognized_ph != '||':
                            segment_index += 1
                
                # Step 2: Execute validation tasks in parallel
                if validation_tasks:
                    print(f"Starting parallel validation of {len(validation_tasks)} phonemes...")
                    # Determine optimal number of workers (8 for M3, but don't exceed task count)
                    max_workers = min(8, len(validation_tasks), os.cpu_count() or 4)
                    
                    validation_results = []
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all tasks
                        future_to_task = {
                            executor.submit(validate_single_phoneme, task): task
                            for task in validation_tasks
                        }
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_task):
                            try:
                                result = future.result()
                                validation_results.append(result)
                            except Exception as e:
                                task = future_to_task[future]
                                print(f"Validation error for phoneme pair {task.get('phoneme_pair', 'unknown')} at index {task.get('index', -1)}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Step 3: Sort results by index to maintain order
                    validation_results.sort(key=lambda x: x['index'])
                    
                    # Step 4: Apply results to aligned_pairs and diagnostic_results
                    for result in validation_results:
                        i = result['index']
                        validation_result = result['validation_result']
                        expected_ph = result['expected_phoneme']
                        recognized_ph = result['recognized_phoneme']
                        
                        # Skip if there was an error
                        if result.get('error'):
                            print(f"Validation error at index {i}: {result['error']}")
                            continue
                        
                        validation_count += 1
                        
                        # Check if validation says it's correct with high confidence
                        is_correct = validation_result.get('is_correct', False)
                        confidence = validation_result.get('confidence', 0.0)
                        
                        if is_correct and confidence > config.VALIDATION_CONFIDENCE_THRESHOLD:
                            print(f"Validation: {expected_ph} -> {recognized_ph} is CORRECT (confidence: {confidence:.2%})")
                            
                            # Update aligned_pairs to mark as correct (for green color)
                            # Change recognized phoneme to match expected
                            aligned_pairs[i] = (expected_ph, expected_ph)
                            
                            # Update diagnostic_results if index matches
                            if i < len(diagnostic_results):
                                diagnostic_results[i]['is_correct'] = True
                                diagnostic_results[i]['validation_result'] = validation_result
                                diagnostic_results[i]['validation_confidence'] = confidence
                                diagnostic_results[i]['validation_override'] = True
                            
                            validation_corrected_count += 1
                        else:
                            # Store validation result but don't override
                            if i < len(diagnostic_results):
                                diagnostic_results[i]['validation_result'] = validation_result
                                diagnostic_results[i]['validation_confidence'] = confidence
                            
                            if is_correct:
                                print(f"Validation: {expected_ph} -> {recognized_ph} is correct but low confidence ({confidence:.2%})")
                            else:
                                print(f"Validation: {expected_ph} -> {recognized_ph} is INCORRECT (confidence: {confidence:.2%})")
                
                print(f"Validation complete: {validation_count} phonemes validated, {validation_corrected_count} corrected")
            
            validation_elapsed = (time.time() - validation_start) * 1000            
            # Stage 10: Visualization
            viz_start = time.time()
            # Output 0: Side-by-side comparison
            viz_side_by_side_start = time.time()
            side_by_side_html = create_side_by_side_comparison(
                expected_phonemes,
                raw_phonemes,
                aligned_pairs
            )
            viz_side_by_side_elapsed = (time.time() - viz_side_by_side_start) * 1000            
            # Output 1: Colored text
            # Convert aligned_pairs to dict format for visualization (for backward compatibility)
            aligned_pairs_dict = []
            for i, (exp, rec) in enumerate(aligned_pairs):
                if i < len(diagnostic_results):
                    aligned_pairs_dict.append(diagnostic_results[i])
                else:
                    aligned_pairs_dict.append({
                        'expected': exp,
                        'recognized': rec,
                        'is_correct': exp == rec,
                        'is_missing': exp is not None and rec is None,
                        'is_extra': exp is None and rec is not None
                    })
            
            # Use recognized text for colored text if original text was empty
            text_for_colored = recognized_text if text_is_empty else text
            # Get expected phonemes dict for the text used for coloring
            # If using recognized text, get phonemes for recognized text, otherwise use original expected_phonemes_dict
            expected_phonemes_dict_for_coloring = expected_phonemes_dict
            if text_is_empty and recognized_text:
                # Re-get expected phonemes for recognized text
                from modules.g2p_module import get_expected_phonemes as get_expected_phonemes_func
                try:
                    expected_phonemes_dict_for_coloring = get_expected_phonemes_func(recognized_text)
                except Exception as e:
                    print(f"Warning: Failed to get expected phonemes for recognized text: {e}")
                    expected_phonemes_dict_for_coloring = expected_phonemes_dict
            
            viz_colored_start = time.time()
            
            # Create colored text for after validation (current state)
            colored_text_html_after = create_colored_text(
                text_for_colored, 
                aligned_pairs_dict,
                expected_phonemes_dict=expected_phonemes_dict_for_coloring,
                aligned_pairs_tuples=aligned_pairs
            )
            
            # If validation is enabled, also create colored text for before validation
            if enable_validation and aligned_pairs_before_validation is not None:
                # Convert aligned_pairs_before_validation to dict format
                aligned_pairs_dict_before = []
                for i, (exp, rec) in enumerate(aligned_pairs_before_validation):
                    if i < len(diagnostic_results_before_validation):
                        aligned_pairs_dict_before.append(diagnostic_results_before_validation[i])
                    else:
                        aligned_pairs_dict_before.append({
                            'expected': exp,
                            'recognized': rec,
                            'is_correct': exp == rec,
                            'is_missing': exp is not None and rec is None,
                            'is_extra': exp is None and rec is not None
                        })
                
                # Create colored text for before validation
                colored_text_html_before = create_colored_text(
                    text_for_colored,
                    aligned_pairs_dict_before,
                    expected_phonemes_dict=expected_phonemes_dict_for_coloring,
                    aligned_pairs_tuples=aligned_pairs_before_validation
                )
                
                # Create comparison view
                colored_text_html = create_validation_comparison(
                    text_for_colored,
                    colored_text_html_before,
                    colored_text_html_after,
                    enable_validation=True
                )
            else:
                # No validation, just use the after version (which is the same as before)
                colored_text_html = colored_text_html_after
            
            viz_colored_elapsed = (time.time() - viz_colored_start) * 1000            
            # Output 2: Detailed report (with WER and PER)
            # Don't show WER if text was empty (WER not calculated)
            show_wer_in_report = config.SHOW_WER and not text_is_empty and wer_result is not None
            text_for_report = recognized_text if text_is_empty else text
            viz_report_start = time.time()
            detailed_report_html = create_detailed_report(
                aligned_pairs_dict,
                diagnostic_results,
                text_for_report,
                wer_result=wer_result if show_wer_in_report else None,
                per_result=per_result if config.SHOW_PER else None,
                recognized_text=recognized_text
            )
            viz_report_elapsed = (time.time() - viz_report_start) * 1000            
            # Output 3: Technical information
            wer_info = ""
            # Only show WER if text was provided (not empty) and WER was calculated
            if not text_is_empty and wer_result and config.SHOW_WER:
                wer_info = f"""
                    <li><strong>WER (Word Error Rate):</strong> {wer_result['wer']:.2%}</li>
                    <li><strong>WER Details:</strong> {wer_result['substitutions']} substitutions, {wer_result['deletions']} deletions, {wer_result['insertions']} insertions</li>
                    <li><strong>Recognized text:</strong> {recognized_text or 'N/A'}</li>
                """
            elif text_is_empty and recognized_text:
                wer_info = f"""
                    <li><strong>Text source:</strong> Extracted from audio (no manual text input)</li>
                    <li><strong>Recognized text:</strong> {recognized_text}</li>
                    <li><strong>WER:</strong> Not calculated (no reference text provided)</li>
                """
            
            per_info = ""
            if per_result and config.SHOW_PER:
                per_info = f"""
                    <li><strong>PER (Phoneme Error Rate):</strong> {per_result['per']:.2%}</li>
                    <li><strong>PER Details:</strong> {per_result['substitutions']} substitutions, {per_result['deletions']} deletions, {per_result['insertions']} insertions</li>
                """
            
            validation_info = ""
            if enable_validation and optional_validator:
                validation_info = f"""
                    <li><strong>Optional validation:</strong> Enabled</li>
                    <li><strong>Validated phonemes:</strong> {validation_count}</li>
                    <li><strong>Corrected by validation:</strong> {validation_corrected_count} (confidence > {config.VALIDATION_CONFIDENCE_THRESHOLD:.0%})</li>
                """
            else:
                validation_info = "<li><strong>Optional validation:</strong> Disabled</li>"
            
            technical_html = f"""
            <div style='padding: 10px; background: #f9f9f9; border-radius: 5px;'>
                <h4>Technical Information</h4>
                <ul>
                    <li><strong>VAD:</strong> Disabled (commented out)</li>
                    <li><strong>ASR:</strong> {'Enabled' if (asr_recognizer and config.ASR_ENABLED) else 'Disabled'}</li>
                    <li><strong>Expected phonemes:</strong> {len(expected_phonemes)}</li>
                    <li><strong>Model:</strong> {config.MODEL_NAME}</li>
                    <li><strong>Raw phonemes:</strong> {len(raw_phonemes)}</li>
                    <li><strong>Filtered phonemes:</strong> {len(recognized_phonemes)}</li>
                    <li><strong>Aligned pairs:</strong> {len(aligned_pairs)}</li>
                    <li><strong>Alignment score:</strong> {alignment_score:.2f}</li>
                    {wer_info}
                    {per_info}
                    {validation_info}
                </ul>
            </div>
            """
            
            # Output 4: Raw phonemes (before filtering)
            viz_raw_start = time.time()
            raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
            viz_raw_elapsed = (time.time() - viz_raw_start) * 1000
            viz_elapsed = (time.time() - viz_start) * 1000            
            total_elapsed = (time.time() - start_time) * 1000
            
            # Build chat message with expandable phoneme details
            phoneme_details_html = f"""
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #0066cc; text-decoration: underline;">Show phonemes</summary>
                <div style="margin-top: 10px;">
                    {side_by_side_html}
                </div>
            </details>
            """
            
            # Create main chat message with colored text and phoneme details
            assistant_message = f"""
            <div style="padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
                {colored_text_html}
                {phoneme_details_html}
            </div>
            """
            
            # Build user message
            user_text = text if text and text.strip() else (recognized_text if recognized_text else "Audio input")
            user_message = f"Text: {user_text}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
            
            # Update chat history
            chat_history = add_to_chat_history(chat_history, user_message, assistant_message)
            
            return (chat_history, text, audio_file)
        
        finally:
            # Clean up temp files (but keep trimmed_audio_path for user to listen)
            # We'll delete it later or let OS clean it up
            if 'tmp_path' in locals():
                Path(tmp_path).unlink(missing_ok=True)
            # Don't delete trimmed_audio_path immediately - user might want to listen to it
            # It will be cleaned up by OS or on next run
            if 'normalized_audio_path' in locals() and normalized_audio_path != tmp_path and normalized_audio_path != trimmed_audio_path:
                Path(normalized_audio_path).unlink(missing_ok=True)
    
    except Exception as e:
        import traceback
        error_html = f"""
        <div style='color: red; padding: 10px; background: #ffe6e6; border-radius: 5px;'>
            <h4>Error occurred:</h4>
            <p>{str(e)}</p>
            <details>
                <summary>Technical details</summary>
                <pre style='font-size: 12px;'>{traceback.format_exc()}</pre>
            </details>
        </div>
        """
        # Update chat history with error
        user_message = f"Text: {text if text else 'Audio input'}" + (f" | Validation: {'Enabled' if enable_validation else 'Disabled'}" if enable_validation else "")
        chat_history = add_to_chat_history(chat_history, user_message, error_html)
        return (chat_history, text, audio_file)


def load_models_in_background():
    """Load models in background: first ASR, then dictionaries, then phoneme model."""
    print("Starting background model loading...")
    try:
        # Stage 1: Load ASR (Whisper or macOS Speech) first - this is critical for user experience
        print("Stage 1: Loading ASR...")
        initialize_asr_only()
        print("ASR loaded successfully!")
        
        # Stage 2: Load dictionaries after ASR is ready
        print("Stage 2: Loading G2P dictionaries...")
        load_dictionaries_in_background()
        
        # Stage 3: Load phoneme recognition model (Wav2Vec2) in background
        # This prevents 5+ second delay on first button click
        load_phoneme_model_in_background()
        
        # Stage 4: Load MFA aligner in background (if enabled)
        if config.MFA_ENABLED:
            load_mfa_in_background()
        
        print("All models loaded successfully in background!")
    except Exception as e:
        print(f"Warning: Some components failed to initialize in background: {e}")


def create_interface():
    """Create Gradio interface."""
    
    # Don't initialize components on startup - let them load in background
    # This allows browser to open quickly
    
    with gr.Blocks(title="German Pronunciation Diagnostic App (L2-Trainer)", theme=gr.themes.Monochrome(), css="""
        /* Center-align the main heading */
        .gradio-container h1 {
            text-align: center !important;
        }
        /* Set font for entire interface */
        .gradio-container, .gradio-container * {
            font-family: 'Consolas', monospace !important;
        }
        .gradio-container .chatbot {
            height: 70vh !important;
            min-height: 400px;
        }
        /* Align elements by height - works for rows with equal_height (excluding unequal-height) */
        .gradio-container .row:not(.unequal-height) > .column {
            display: flex !important;
            align-items: stretch !important;
        }
        .gradio-container .row:not(.unequal-height) > .column > .block {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
        }
        /* Align text field by height - occupies entire available area (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child .form {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        /* Textarea container occupies full height (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block > label {
            flex: 1 !important;
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:first-child .form .block > label > .input-container {
            flex: 1 !important;
            display: flex !important;
            height: 100% !important;
        }
        /* Textarea occupies full height of block (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:first-child textarea {
            flex: 1 !important;
            height: 100% !important;
            min-height: 100% !important;
            resize: none !important;
        }
        /* Align checkbox and button to right edge in one column (only for equal_height rows) */
        .gradio-container .row:not(.unequal-height) > .column:last-child {
            justify-content: flex-end !important;
            align-items: flex-end !important;
        }
        .gradio-container .row:not(.unequal-height) > .column:last-child .block {
            display: flex !important;
            flex-direction: column !important;
            align-items: flex-end !important;
            gap: 10px !important;
        }
        /* Improve display of phoneme sequences */
        /* Phoneme containers should use full available width */
        .gradio-container div[data-block-id='side-by-side-comparison'] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        /* Phoneme sequences should be distributed across full width */
        .gradio-container div[data-block-id='side-by-side-comparison'] > div > div {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }
        /* Natural wrapping of phoneme sequences, like regular text */
        /* Phoneme containers should use natural line wrapping, like regular text */
        .gradio-container div[data-block-id='side-by-side-comparison'] div[style*="font-size: 18px"][style*="line-height: 1.3"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        /* Inline-block elements should wrap naturally, like regular text */
        .gradio-container div[data-block-id='side-by-side-comparison'] div[style*="font-size: 18px"] > span[style*="display: inline-block"] {
            white-space: normal !important;
            overflow-wrap: break-word !important;
        }
        /* Horizontal scrolling for long sequences */
        .gradio-container div[style*="overflow-x: auto"] {
            scrollbar-width: thin !important;
            scrollbar-color: #cbd5e0 #f7fafc !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar {
            height: 6px !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar-track {
            background: #f7fafc !important;
        }
        .gradio-container div[style*="overflow-x: auto"]::-webkit-scrollbar-thumb {
            background: #cbd5e0 !important;
            border-radius: 3px !important;
        }
        /* Adaptive distribution for different screen sizes */
        @media (min-width: 1200px) {
            .gradio-container div[data-block-id='side-by-side-comparison'] {
                max-width: calc(100vw - 200px) !important;
            }
        }
        @media (max-width: 768px) {
            .gradio-container div[data-block-id='side-by-side-comparison'] {
                max-width: calc(100vw - 40px) !important;
            }
        }
        /* Stretch main Row with chat and controls */
        .gradio-container .row.unequal-height {
            display: flex !important;
            align-items: stretch !important;
            min-height: 600px !important;
        }
        /* Stretch left column with chat */
        .gradio-container .row.unequal-height > .column:first-child {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
        }
        /* Stretch right column to chat height */
        .gradio-container .row.unequal-height > .column:nth-child(2) {
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            align-items: stretch !important;
        }
        /* Stretch inner Row with controls by height */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row {
            display: flex !important;
            flex: 1 !important;
            align-items: stretch !important;
            min-height: 0 !important;
        }
        /* Align Audio Input to top edge */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:first-child {
            display: flex !important;
            align-items: flex-start !important;
        }
        /* Align validation controls to bottom edge - stretch to full height */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            align-items: stretch !important;
        }
        /* Block with validation controls - aligns button to bottom */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child > .block {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            gap: 10px !important;
            flex: 1 !important;
        }
        /* Form with validation controls - aligns button to bottom */
        .gradio-container .row.unequal-height > .column:nth-child(2) > .row > .column:last-child > .form {
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-end !important;
            flex: 1 !important;
        }
    """) as app:
        gr.Markdown("""
        # German Pronunciation Diagnostic App (L2-Trainer)
        """)
        
        # Main layout: Chatbot on left (70%), controls on right (30%)
        with gr.Row():
            # Left side: Chatbot (70% width)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Pronunciation Analysis",
                    height=600,
                    show_label=False,
                    container=True
                )
            
            # Right side: Controls (30% width)
            with gr.Column(scale=3):
                # German Text input (full width, 1.5x height)
                text_input = gr.Textbox(
                    label="German Text",
                    placeholder="Enter a German sentence here...",
                    lines=6,  # Increased from 2 to 3 (1.5x height)
                    show_label=True
                )
                
                # Audio Input and validation controls in one row
                with gr.Row():
                    # Audio Input (narrower, 2x narrower than validation controls, aligned to top of chat)
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Audio Input",
                            type="numpy",
                            sources=["microphone", "upload"],
                            show_label=True
                        )
                    
                    # Validation controls (wider, right side)
                    with gr.Column(scale=2):
                        validation_checkbox = gr.Checkbox(
                            label="Enable 2 step validation",
                            value=False,
                            show_label=True
                        )
                        process_btn = gr.Button("Validate Pronunciation", variant="primary", size="sm")
        
        # Examples - first row
        gr.Examples(
            examples=[
                ["Hallo, wie geht es dir?", None],
                ["Ich habe einen Apfel.", None],
                ["Der Bär trägt einen Ball.", None],
                ["Ich möchte ein Stück Kuchen.", None],
            ],
            inputs=[text_input, audio_input]
        )
        
        # Custom button for loading text and audio
        def load_example_text_and_audio():
            """Load example text and audio file."""
            text = "Im Grundlagenstreit der Mathematik entspräche der nominalistischen Position die formalistische Richtung."
            audio_path = "/Volumes/SSanDisk/audio_data/data_wav/TV-2021.02-Neutral/4aeeae88-0777-2c8c-5c93-2e844a462e49---7c5cf6a7351fb3ca39004d5e49566c09.wav"
            
            # Load audio file
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                # Gradio expects (sample_rate, audio_array) tuple
                audio_tuple = (sample_rate, audio_array)
                return text, audio_tuple
            except Exception as e:
                print(f"Error loading audio file: {e}")
                # Return text only if audio fails to load
                return text, None
        
        # Function for second example
        def load_example_text_and_audio_2():
            """Load example text and audio file (example 2)."""
            text = """Aber für unsere Entwicklungspolitik, für unsere Außenpolitik, für unsere Kulturpolitik durch die Goethe-Institute ist das Thema „Teilhabe von Frauen" ein zentrales Thema."""
            audio_path = "/Volumes/SSanDisk/audio_data/data_wav/TV-2021.02-Neutral/4aeeae88-0777-2c8c-5c93-2e844a462e49---0a05b797c25f88e74d0d8d69a4705187.wav"
            
            # Load audio file
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                # Gradio expects (sample_rate, audio_array) tuple
                audio_tuple = (sample_rate, audio_array)
                return text, audio_tuple
            except Exception as e:
                print(f"Error loading audio file: {e}")
                # Return text only if audio fails to load
                return text, None
        
        # Function for third example
        def load_example_text_and_audio_3():
            """Load example text and audio file (example 3)."""
            text = "Plötzlich wurde dem Privatdetektiv klar, worum es dem Dieb eigentlich ging."
            audio_path = "/Volumes/SSanDisk/SpeechRec-German/wav2vec2-finetune/artifacts/processed_audio/TV-2021.02-Neutral/4aeeae88-0777-2c8c-5c93-2e844a462e49---8da112ef2540faff1fe1dfdf3f433e54.wav"
            
            # Load audio file
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                # Gradio expects (sample_rate, audio_array) tuple
                audio_tuple = (sample_rate, audio_array)
                return text, audio_tuple
            except Exception as e:
                print(f"Error loading audio file: {e}")
                # Return text only if audio fails to load
                return text, None
        
        # Second row of examples with custom buttons
        with gr.Row(equal_height=True):
            example_btn = gr.Button("Im Grundlagenstreit der...", variant="secondary", size="sm")
            example_btn.click(
                fn=load_example_text_and_audio,
                inputs=[],
                outputs=[text_input, audio_input]
            )
            
            example_btn_2 = gr.Button("Aber für unsere Entwicklungspolitik...", variant="secondary", size="sm")
            example_btn_2.click(
                fn=load_example_text_and_audio_2,
                inputs=[],
                outputs=[text_input, audio_input]
            )
            
            example_btn_3 = gr.Button("Plötzlich wurde dem Privatdetektiv...", variant="secondary", size="sm")
            example_btn_3.click(
                fn=load_example_text_and_audio_3,
                inputs=[],
                outputs=[text_input, audio_input]
            )
        
        # Process button
        process_btn.click(
            fn=process_pronunciation,
            inputs=[text_input, audio_input, validation_checkbox, chatbot],
            outputs=[chatbot, text_input, audio_input]
        )
        
        gr.Markdown("""
        ### Instructions:
        1. Enter a German sentence in the text box
        2. Record audio using the microphone or upload an audio file (WAV, 16kHz recommended)
        3. Optionally enable additional validation through trained models
        4. Click "Validate Pronunciation" to see results
        """)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    
    # Start background model loading in a separate thread
    # This happens while the server is starting up
    background_thread = threading.Thread(target=load_models_in_background, daemon=True)
    background_thread.start()
    
    # Launch the interface
    # Browser will open quickly, models will load in background
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


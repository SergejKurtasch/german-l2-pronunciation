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
from typing import List, Dict, Optional, Tuple

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

# Global instances
# vad_detector = None  # VAD disabled
audio_normalizer = None
phoneme_recognizer = None
phoneme_filter = None
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
            else:                print(f"Warning: ASR recognizer not available (neither {requested_engine} nor Whisper available)")
        except Exception as e:            print(f"Warning: ASR recognizer initialization failed: {e}")
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
            phoneme_recognizer = get_phoneme_recognizer(
                model_name=config.MODEL_NAME,
                device=config.MODEL_DEVICE if config.MODEL_DEVICE != "auto" else None
            )
            model_load_elapsed = (time.time() - model_load_start) * 1000            print(f"Phoneme recognition model loaded successfully! (took {model_load_elapsed/1000:.2f}s)")
        else:            print("Phoneme recognition model already loaded.")
    except Exception as e:        print(f"Warning: Phoneme model loading failed: {e}")


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
    global audio_normalizer, phoneme_recognizer, phoneme_filter, forced_aligner, diagnostic_engine, optional_validator, asr_recognizer
    
    import json, time
    init_components_start = time.time()
    
    if audio_normalizer is None:
        try:
            comp_start = time.time()
            audio_normalizer = get_audio_normalizer()
            comp_elapsed = (time.time() - comp_start) * 1000            print("Audio normalizer initialized")
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
            comp_elapsed = (time.time() - comp_start) * 1000            print(f"Phoneme recognizer (Wav2Vec2 XLSR-53 eSpeak) initialized with model: {phoneme_recognizer.model_name}")
        except Exception as e:
            print(f"Error: Phoneme recognizer initialization failed: {e}")
            raise
    
    if phoneme_filter is None:
        comp_start = time.time()
        phoneme_filter = get_phoneme_filter(
            whitelist=config.PHONEME_WHITELIST,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        comp_elapsed = (time.time() - comp_start) * 1000        print("Phoneme filter initialized")
    
    if forced_aligner is None:
        comp_start = time.time()
        forced_aligner = get_forced_aligner(blank_id=config.FORCED_ALIGNMENT_BLANK_ID)
        comp_elapsed = (time.time() - comp_start) * 1000        print("Forced aligner initialized")
    
    if diagnostic_engine is None:
        comp_start = time.time()
        diagnostic_engine = get_diagnostic_engine()
        comp_elapsed = (time.time() - comp_start) * 1000        print("Diagnostic engine initialized")
    
    if optional_validator is None:
        comp_start = time.time()
        optional_validator = get_optional_validator()
        comp_elapsed = (time.time() - comp_start) * 1000        print("Optional validator initialized")
    
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


def process_pronunciation(
    text: str,
    audio_file: Optional[Tuple[int, np.ndarray]] = None,
    enable_validation: bool = False
) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    Process pronunciation validation.
    
    Args:
        text: German text input
        audio_file: Tuple of (sample_rate, audio_array) from Gradio
        enable_validation: Whether to enable optional validation through trained models
        
    Returns:
        Tuple of:
        1. Text with sources (HTML)
        2. Expected phonemes (HTML)
        3. Recognized phonemes (HTML)
        4. Side-by-side comparison (HTML)
        5. Colored text (HTML)
        6. Detailed report (HTML)
        7. Technical information (HTML)
        8. Raw phonemes (before filtering) (HTML)
    """    
    # Check if text is empty - if so, we'll use ASR to get text from audio
    text_is_empty = not text or not text.strip()
    
    if audio_file is None:
        error_html = "<div style='color: orange; padding: 10px;'>Please record or upload audio.</div>"
        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
    
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
            return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
        
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
                                asr_device = "unknown"                        print(f"ASR: Recognized text (from audio): '{recognized_text}'")
                        
                        if not recognized_text or not recognized_text.strip():
                            error_html = "<div style='color: orange; padding: 10px;'>Could not recognize text from audio. Please try again or enter text manually.</div>"
                            return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
                    except Exception as e:
                        print(f"Error: ASR failed: {e}")
                        error_html = f"<div style='color: red; padding: 10px;'>Failed to recognize text from audio: {str(e)}</div>"
                        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
                else:
                    error_html = "<div style='color: orange; padding: 10px;'>ASR is not available. Please enter text manually or enable ASR in configuration.</div>"
                    return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
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
                            asr_device = "unknown"                    print(f"ASR: Recognized text: '{recognized_text}'")
                    
                    # Stage 3: WER Calculation (only if text was provided)
                    if recognized_text:
                        wer_start = time.time()
                        try:
                            wer_result = calculate_wer(text, recognized_text)
                            wer_elapsed = (time.time() - wer_start) * 1000                            print(f"WER: {wer_result['wer']:.2%} (Substitutions: {wer_result['substitutions']}, "
                                  f"Deletions: {wer_result['deletions']}, Insertions: {wer_result['insertions']})")
                        except Exception as e:
                            wer_elapsed = (time.time() - wer_start) * 1000                            print(f"Error: WER calculation failed: {e}")
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
            if not text_is_empty and wer_result and wer_result['wer'] > config.WER_THRESHOLD and config.WER_SKIP_PHONEME_ANALYSIS:                # High WER - show only text comparison
                from modules.visualization import create_text_comparison_view
                
                # Create simplified view
                try:                    comparison_html = create_text_comparison_view(text, recognized_text or "", wer_result)                except Exception as e:                    print(f"Error: Failed to create text comparison view: {e}")
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
                
                # Create text with sources display even for high WER case
                from modules.visualization import create_text_with_sources_display
                # For high WER, we still want to show the expected text with sources
                # Use recognized text for phonemes if available, otherwise use expected text
                text_for_sources = recognized_text if recognized_text else text
                # Get expected phonemes for display (even though we skip detailed analysis)
                # Import get_expected_phonemes here to avoid any scope issues
                from modules.g2p_module import get_expected_phonemes as get_expected_phonemes_func
                try:
                    expected_phonemes_dict_for_sources = get_expected_phonemes_func(text_for_sources)
                    text_with_sources_html = create_text_with_sources_display(
                        text_for_sources,
                        expected_phonemes_dict_for_sources
                    )
                except Exception as e:
                    # Fallback if get_expected_phonemes fails
                    print(f"Warning: Failed to get expected phonemes for text with sources: {e}")
                    import traceback
                    traceback.print_exc()
                    text_with_sources_html = f"<div style='padding: 15px; background: #f8f9fa; border-radius: 5px;'><p>Original Text: {text_for_sources}</p></div>"
                
                return (
                    text_with_sources_html,  # First output: text with sources
                    empty_html,
                    empty_html,
                    comparison_html,
                    comparison_html,
                    comparison_html,
                    technical_html,
                    raw_phonemes_html
                )
            
            # Stage 5: G2P - Get phonemes from recognized text (or expected text if ASR not available)
            # Use recognized text for phoneme analysis if available, otherwise use expected text
            g2p_start = time.time()
            text_for_phonemes = recognized_text if recognized_text else text
            expected_phonemes_dict = get_expected_phonemes(text_for_phonemes)
            expected_phonemes = [ph.get('phoneme', '') for ph in expected_phonemes_dict]
            # Apply CTC collapse logic to expected phonemes (same as model does)
            expected_phonemes = collapse_consecutive_duplicates(expected_phonemes)
            g2p_elapsed = (time.time() - g2p_start) * 1000            print(f"Expected phonemes (from {'recognized' if recognized_text else 'expected'} text): {len(expected_phonemes)}")
            
            if not expected_phonemes:
                error_html = "<div style='color: red; padding: 10px;'>Failed to extract expected phonemes from text.</div>"
                return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)
            
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
            decode_elapsed = (time.time() - decode_start) * 1000            phoneme_rec_elapsed = (time.time() - phoneme_rec_start) * 1000            
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
            filter_elapsed = (time.time() - filter_start) * 1000            print(f"Filtered phonemes: {len(recognized_phonemes)}")
            
            # Use raw_phonemes for validation - model already outputs accurate IPA phonemes
            if not raw_phonemes:
                error_html = "<div style='color: orange; padding: 10px;'>No phonemes recognized. Audio may be unclear.</div>"
                # Create raw phonemes display even if filtered is empty
                raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
                return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, raw_phonemes_html)
            
            # Stage 5: Forced Alignment (for recognized phonemes)
            # Load waveform for forced alignment
            alignment_start = time.time()
            waveform_load_start = time.time()
            waveform, sr = librosa.load(trimmed_audio_path, sr=config.SAMPLE_RATE, mono=True)
            waveform_load_elapsed = (time.time() - waveform_load_start) * 1000            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            
            # Extract segments for recognized phonemes
            # Use ARPABET phonemes for forced alignment (vocab contains ARPABET tokens)
            recognized_phonemes_arpabet = [
                ph.get('phoneme_arpabet', ph.get('phoneme', '')) 
                for ph in filtered_phonemes 
                if ph.get('phoneme_arpabet') or ph.get('phoneme')
            ]
            recognized_segments = []
            if len(recognized_phonemes_arpabet) > 0:
                try:
                    forced_align_start = time.time()
                    recognized_segments = forced_aligner.extract_phoneme_segments(
                        waveform_tensor,
                        recognized_phonemes_arpabet,
                        emissions,
                        vocab,
                        config.SAMPLE_RATE
                    )
                    forced_align_elapsed = (time.time() - forced_align_start) * 1000                    # Update segment labels to IPA - use raw_phonemes for accurate IPA labels
                    # Note: segment count may differ from raw_phonemes count due to forced alignment
                    for i, segment in enumerate(recognized_segments):
                        if i < len(raw_phonemes):
                            segment.label = raw_phonemes[i]
                except Exception as e:
                    print(f"Warning: Forced alignment failed: {e}")
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
            per_elapsed = (time.time() - per_start) * 1000            print(f"PER: {per_result['per']:.2%} (Substitutions: {per_result['substitutions']}, "
                  f"Deletions: {per_result['deletions']}, Insertions: {per_result['insertions']})")
            
            # Stage 8: Diagnostic Analysis
            diagnostic_start = time.time()
            diagnostic_results = diagnostic_engine.analyze_pronunciation(aligned_pairs)
            diagnostic_elapsed = (time.time() - diagnostic_start) * 1000            
            # Store aligned_pairs before validation for comparison
            aligned_pairs_before_validation = [(exp, rec) for exp, rec in aligned_pairs] if enable_validation else None
            diagnostic_results_before_validation = [dict(dr) for dr in diagnostic_results] if enable_validation else None
            
            # Stage 9: Optional Validation
            validation_start = time.time()
            validation_count = 0
            validation_corrected_count = 0
            if enable_validation and optional_validator:
                # For each mismatch in aligned_pairs, try to validate with trained model
                print(f"Optional validation enabled - checking {len(aligned_pairs)} aligned pairs")                
                # Build index mapping from recognized phoneme to segments
                # This helps when multiple segments have the same phoneme
                segment_index = 0
                
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
                            audio_segment = waveform[start_sample:end_sample]
                            
                            # Fallback for empty or very short segments (< 100 samples = ~6ms)
                            # This happens when forced aligner fails to determine boundaries (e.g., at end of audio)
                            MIN_SEGMENT_LENGTH = 100  # samples
                            CONTEXT_MS = 100.0  # Use 100ms context window
                            
                            if len(audio_segment) < MIN_SEGMENT_LENGTH:                                
                                # Use segment start_time as center point, or estimate from index
                                center_time = segment.start_time if segment.start_time > 0 else (segment_index / len(recognized_segments)) * (len(waveform) / config.SAMPLE_RATE)
                                
                                # Extract context window around the position
                                context_samples = int(CONTEXT_MS / 1000 * config.SAMPLE_RATE)
                                half_context = context_samples // 2
                                center_sample = int(center_time * config.SAMPLE_RATE)
                                
                                fallback_start = max(0, center_sample - half_context)
                                fallback_end = min(len(waveform), center_sample + half_context)
                                audio_segment = waveform[fallback_start:fallback_end]                            
                            # Validate
                            validation_result = optional_validator.validate_phoneme_segment(
                                audio_segment,
                                phoneme_pair=phoneme_pair,
                                expected_phoneme=expected_ph,
                                suspected_phoneme=recognized_ph,
                                sr=config.SAMPLE_RATE
                            )                            
                            validation_count += 1
                            
                            # Check if validation says it's correct with high confidence (>70%)
                            is_correct = validation_result.get('is_correct', False)
                            confidence = validation_result.get('confidence', 0.0)
                            
                            if is_correct and confidence > 0.7:
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
                        else:
                            print(f"Warning: No segment found for {recognized_ph} at index {segment_index}")
                        
                        # Advance segment index
                        segment_index += 1
                    else:
                        # No model for this pair, advance segment index
                        if recognized_ph is not None and recognized_ph != '||':
                            segment_index += 1
                
                print(f"Validation complete: {validation_count} phonemes validated, {validation_corrected_count} corrected")
            
            validation_elapsed = (time.time() - validation_start) * 1000            
            # Stage 10: Visualization
            viz_start = time.time()
            # Output 0: Text with source information (for debugging)
            from modules.visualization import create_text_with_sources_display
            viz_text_sources_start = time.time()
            text_with_sources_html = create_text_with_sources_display(
                text_for_phonemes,
                expected_phonemes_dict
            )
            viz_text_sources_elapsed = (time.time() - viz_text_sources_start) * 1000            
            # Output 1: Expected phonemes (show expected phonemes directly, with spaces between phonemes)            expected_phonemes_str = ' '.join(expected_phonemes)            expected_html = f"<div style='font-family: monospace; font-size: 14px;'><p>{expected_phonemes_str}</p></div>"
            
            # Output 2: Recognized phonemes
            # Use raw_phonemes directly - model already outputs accurate IPA phonemes without filtering            
            viz_recognized_start = time.time()
            recognized_html = create_simple_phoneme_comparison([], raw_phonemes)
            viz_recognized_elapsed = (time.time() - viz_recognized_start) * 1000            
            # Output 3: Side-by-side comparison
            viz_side_by_side_start = time.time()
            side_by_side_html = create_side_by_side_comparison(
                expected_phonemes,
                raw_phonemes,
                aligned_pairs
            )
            viz_side_by_side_elapsed = (time.time() - viz_side_by_side_start) * 1000            
            # Output 4: Colored text
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
            # Output 5: Detailed report (with WER and PER)
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
            # Output 6: Technical information
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
                    <li><strong>Corrected by validation:</strong> {validation_corrected_count} (confidence > 70%)</li>
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
            
            # Output 7: Raw phonemes (before filtering)
            viz_raw_start = time.time()
            raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
            viz_raw_elapsed = (time.time() - viz_raw_start) * 1000            viz_elapsed = (time.time() - viz_start) * 1000            
            total_elapsed = (time.time() - start_time) * 1000            
            return (
                text_with_sources_html,  # First output: text with sources
                expected_html,
                recognized_html,
                side_by_side_html,
                colored_text_html,
                detailed_report_html,
                technical_html,
                raw_phonemes_html
            )
        
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
        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html)


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
        
        print("All models loaded successfully in background!")
    except Exception as e:
        print(f"Warning: Some components failed to initialize in background: {e}")


def create_interface():
    """Create Gradio interface."""
    
    # Don't initialize components on startup - let them load in background
    # This allows browser to open quickly
    
    with gr.Blocks(title="German Pronunciation Diagnostic App (L2-Trainer)", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # German Pronunciation Diagnostic App (L2-Trainer)
        
        Enter a German sentence and record your pronunciation.
        """)
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="German Text",
                    placeholder="Enter a German sentence here...",
                    lines=3
                )
                
                audio_input = gr.Audio(
                    label="Record or Upload Audio",
                    type="numpy",
                    sources=["microphone", "upload"]
                )
                
                validation_checkbox = gr.Checkbox(
                    label="Enable optional validation through trained models",
                    value=False,
                    info="Increases processing time but improves accuracy for incorrect phonemes"
                )
                
                process_btn = gr.Button("Validate Pronunciation", variant="primary")
            
            with gr.Column():
                text_with_sources_output = gr.HTML(label="0. Original Text with Transcription Sources")
                expected_output = gr.HTML(label="1. Expected Phonemes")
                recognized_output = gr.HTML(label="2. Recognized Phonemes")
                comparison_output = gr.HTML(label="3. Side-by-Side Comparison")
                colored_output = gr.HTML(label="4. Colored Text")
                report_output = gr.HTML(label="5. Detailed Report")
                technical_output = gr.HTML(label="6. Technical Information")
                raw_phonemes_output = gr.HTML(label="7. Raw Phonemes (Before Filtering)")
        
        # Examples
        gr.Examples(
            examples=[
                ["Hallo, wie geht es dir?", None],
                ["Ich habe einen Apfel.", None],
                ["Das Wetter ist schön heute.", None],
                ["Der Bär trägt einen Ball.", None],
                ["Ich möchte ein Stück Kuchen.", None],
            ],
            inputs=[text_input, audio_input]
        )
        
        # Process button
        process_btn.click(
            fn=process_pronunciation,
            inputs=[text_input, audio_input, validation_checkbox],
            outputs=[
                text_with_sources_output,
                expected_output,
                recognized_output,
                comparison_output,
                colored_output,
                report_output,
                technical_output,
                raw_phonemes_output
            ]
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


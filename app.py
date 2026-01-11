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
    create_raw_phonemes_display
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
    """Initialize only ASR (Whisper) for fast startup."""
    global asr_recognizer
    
    if asr_recognizer is None and config.ASR_ENABLED:
        try:
            asr_recognizer = get_speech_recognizer(
                model_size=config.ASR_MODEL,
                device=config.ASR_DEVICE
            )
            if asr_recognizer:
                print(f"ASR recognizer (Whisper {config.ASR_MODEL}) initialized")
            else:
                print("Warning: ASR recognizer not available (whisper not installed)")
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


def initialize_components():
    """Initialize global components."""
    # global vad_detector, audio_normalizer, phoneme_recognizer, phoneme_filter, forced_aligner, diagnostic_engine, optional_validator, asr_recognizer
    global audio_normalizer, phoneme_recognizer, phoneme_filter, forced_aligner, diagnostic_engine, optional_validator, asr_recognizer
    
    if audio_normalizer is None:
        try:
            audio_normalizer = get_audio_normalizer()
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
            phoneme_recognizer = get_phoneme_recognizer(
                model_name=config.MODEL_NAME,
                device=config.MODEL_DEVICE if config.MODEL_DEVICE != "auto" else None
            )
            print(f"Phoneme recognizer (Wav2Vec2 XLSR-53 eSpeak) initialized with model: {phoneme_recognizer.model_name}")
        except Exception as e:
            print(f"Error: Phoneme recognizer initialization failed: {e}")
            raise
    
    if phoneme_filter is None:
        phoneme_filter = get_phoneme_filter(
            whitelist=config.PHONEME_WHITELIST,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        print("Phoneme filter initialized")
    
    if forced_aligner is None:
        forced_aligner = get_forced_aligner(blank_id=config.FORCED_ALIGNMENT_BLANK_ID)
        print("Forced aligner initialized")
    
    if diagnostic_engine is None:
        diagnostic_engine = get_diagnostic_engine()
        print("Diagnostic engine initialized")
    
    if optional_validator is None:
        optional_validator = get_optional_validator()
        print("Optional validator initialized")
    
    # ASR is loaded separately in initialize_asr_only()
    # Initialize ASR here only if not already loaded
    if asr_recognizer is None and config.ASR_ENABLED:
        initialize_asr_only()


def process_pronunciation(
    text: str,
    audio_file: Optional[Tuple[int, np.ndarray]] = None,
    enable_validation: bool = False
) -> Tuple[str, str, str, str, str, str, str, Optional[Tuple[int, np.ndarray]]]:
    """
    Process pronunciation validation.
    
    Args:
        text: German text input
        audio_file: Tuple of (sample_rate, audio_array) from Gradio
        enable_validation: Whether to enable optional validation through trained models
        
    Returns:
        Tuple of:
        1. Expected phonemes (HTML)
        2. Recognized phonemes (HTML)
        3. Side-by-side comparison (HTML)
        4. Colored text (HTML)
        5. Detailed report (HTML)
        6. Technical information (HTML)
        7. Raw phonemes (before filtering) (HTML)
        8. Trimmed audio (Tuple[sample_rate, audio_array] or None)
    """
    # Check if text is empty - if so, we'll use ASR to get text from audio
    text_is_empty = not text or not text.strip()
    
    if audio_file is None:
        error_html = "<div style='color: orange; padding: 10px;'>Please record or upload audio.</div>"
        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
    
    try:
        # Initialize components
        initialize_components()
        
        # Extract audio
        if isinstance(audio_file, tuple):
            sample_rate, audio_array = audio_file
        else:
            error_html = "<div style='color: red; padding: 10px;'>Invalid audio format.</div>"
            return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio_array, sample_rate)
        
        try:
            # Stage 0: Audio normalization (for AGC issues)
            normalized_audio_path = tmp_path
            if audio_normalizer is not None and config.ENABLE_AUDIO_NORMALIZATION:
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as normalized_file:
                        normalized_path = normalized_file.name
                    normalized_audio_path = audio_normalizer.process_audio_file(
                        tmp_path,
                        normalized_path,
                        sample_rate,
                        compress_peaks=config.NORMALIZE_COMPRESS_PEAKS,
                        peak_compression_ratio=config.NORMALIZE_PEAK_COMPRESSION_RATIO,
                        peak_compression_duration_ms=config.NORMALIZE_PEAK_COMPRESSION_DURATION_MS,
                        normalize_method=config.NORMALIZE_METHOD
                    )
                    print(f"Audio normalization: Compressed peaks and normalized")
                except Exception as e:
                    print(f"Warning: Audio normalization failed: {e}")
                    normalized_audio_path = tmp_path
            
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
                        print(f"ASR: Recognized text (from audio): '{recognized_text}'")
                        
                        if not recognized_text or not recognized_text.strip():
                            error_html = "<div style='color: orange; padding: 10px;'>Could not recognize text from audio. Please try again or enter text manually.</div>"
                            return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
                    except Exception as e:
                        print(f"Error: ASR failed: {e}")
                        error_html = f"<div style='color: red; padding: 10px;'>Failed to recognize text from audio: {str(e)}</div>"
                        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
                else:
                    error_html = "<div style='color: orange; padding: 10px;'>ASR is not available. Please enter text manually or enable ASR in configuration.</div>"
                    return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
            elif asr_recognizer and config.ASR_ENABLED:
                # Text is provided, ASR is optional (for comparison)
                try:
                    recognized_text = asr_recognizer.transcribe(
                        trimmed_audio_path,
                        language=config.ASR_LANGUAGE
                    )
                    print(f"ASR: Recognized text: '{recognized_text}'")
                    
                    # Stage 3: WER Calculation (only if text was provided)
                    if recognized_text:
                        wer_result = calculate_wer(text, recognized_text)
                        print(f"WER: {wer_result['wer']:.2%} (Substitutions: {wer_result['substitutions']}, "
                              f"Deletions: {wer_result['deletions']}, Insertions: {wer_result['insertions']})")
                except Exception as e:
                    print(f"Warning: ASR failed: {e}")
                    recognized_text = None
                    wer_result = None
            
            # Stage 4: Check WER threshold - skip phoneme analysis if WER is too high
            # Skip this check if text was empty (WER not calculated)
            if not text_is_empty and wer_result and wer_result['wer'] > config.WER_THRESHOLD and config.WER_SKIP_PHONEME_ANALYSIS:
                # High WER - show only text comparison
                from modules.visualization import create_text_comparison_view
                
                # Load trimmed audio for return
                trimmed_audio_data = None
                if trimmed_audio_path and Path(trimmed_audio_path).exists():
                    try:
                        trimmed_audio, trimmed_sr = sf.read(trimmed_audio_path)
                        if len(trimmed_audio.shape) > 1:
                            trimmed_audio = np.mean(trimmed_audio, axis=1)
                        trimmed_audio_data = (trimmed_sr, trimmed_audio)
                    except Exception as e:
                        print(f"Warning: Failed to load trimmed audio for output: {e}")
                
                # Create simplified view
                comparison_html = create_text_comparison_view(text, recognized_text or "", wer_result)
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
                
                return (
                    empty_html,
                    empty_html,
                    comparison_html,
                    comparison_html,
                    comparison_html,
                    technical_html,
                    raw_phonemes_html,
                    trimmed_audio_data
                )
            
            # Stage 5: G2P - Get phonemes from recognized text (or expected text if ASR not available)
            # Use recognized text for phoneme analysis if available, otherwise use expected text
            text_for_phonemes = recognized_text if recognized_text else text
            expected_phonemes_dict = get_expected_phonemes(text_for_phonemes)
            expected_phonemes = [ph.get('phoneme', '') for ph in expected_phonemes_dict]
            print(f"Expected phonemes (from {'recognized' if recognized_text else 'expected'} text): {len(expected_phonemes)}")
            
            if not expected_phonemes:
                error_html = "<div style='color: red; padding: 10px;'>Failed to extract expected phonemes from text.</div>"
                return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)
            
            # Stage 3: Phoneme Recognition (Wav2Vec2 XLSR-53 eSpeak)
            logits, emissions = phoneme_recognizer.recognize_phonemes(
                trimmed_audio_path,
                sample_rate=config.SAMPLE_RATE
            )
            vocab = phoneme_recognizer.get_vocab()
            
            # Decode phonemes (for display)
            decoded_phonemes_str = phoneme_recognizer.decode_phonemes(logits)
            
            # #region agent log
            import json
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"app.py:322","message":"decoded_phonemes_str after decode_phonemes","data":{"decoded_phonemes_str":decoded_phonemes_str,"length":len(decoded_phonemes_str)},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            raw_phonemes = decoded_phonemes_str.split()
            
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"app.py:323","message":"raw_phonemes after split","data":{"raw_phonemes":raw_phonemes,"count":len(raw_phonemes),"has_dash":any('-' in ph for ph in raw_phonemes)},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            print(f"Raw phonemes: {len(raw_phonemes)}")
            
            # Stage 4: Multi-level Filtering
            # Note: Filtering is kept for forced alignment (uses ARPABET), but raw_phonemes are used for display and alignment
            filtered_phonemes = phoneme_filter.filter_combined(
                logits,
                raw_phonemes,
                vocab
            )
            
            recognized_phonemes = [ph.get('phoneme', '') for ph in filtered_phonemes]
            print(f"Filtered phonemes: {len(recognized_phonemes)}")
            
            # Use raw_phonemes for validation - model already outputs accurate IPA phonemes
            if not raw_phonemes:
                error_html = "<div style='color: orange; padding: 10px;'>No phonemes recognized. Audio may be unclear.</div>"
                # Still return trimmed audio even if no phonemes recognized
                trimmed_audio_data = None
                if trimmed_audio_path and Path(trimmed_audio_path).exists():
                    try:
                        trimmed_audio, trimmed_sr = sf.read(trimmed_audio_path)
                        trimmed_audio_data = (trimmed_sr, trimmed_audio)
                    except:
                        pass
                # Create raw phonemes display even if filtered is empty
                raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
                return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, raw_phonemes_html, trimmed_audio_data)
            
            # Stage 5: Forced Alignment (for recognized phonemes)
            # Load waveform for forced alignment
            waveform, sr = librosa.load(trimmed_audio_path, sr=config.SAMPLE_RATE, mono=True)
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            
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
                    recognized_segments = forced_aligner.extract_phoneme_segments(
                        waveform_tensor,
                        recognized_phonemes_arpabet,
                        emissions,
                        vocab,
                        config.SAMPLE_RATE
                    )
                    # Update segment labels to IPA - use raw_phonemes for accurate IPA labels
                    # Note: segment count may differ from raw_phonemes count due to forced alignment
                    for i, segment in enumerate(recognized_segments):
                        if i < len(raw_phonemes):
                            segment.label = raw_phonemes[i]
                except Exception as e:
                    print(f"Warning: Forced alignment failed: {e}")
            
            # Stage 6: Needleman-Wunsch Alignment
            # Use raw_phonemes directly - model already outputs accurate IPA phonemes
            aligned_pairs, alignment_score = needleman_wunsch_align(
                expected_phonemes,
                raw_phonemes,
                match_score=config.NW_MATCH_SCORE,
                mismatch_score=config.NW_MISMATCH_SCORE,
                gap_penalty=config.NW_GAP_PENALTY
            )
            
            # #region agent log
            aligned_pairs_with_dash = [(exp, rec) for exp, rec in aligned_pairs if exp == '-' or rec == '-']
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"app.py:405","message":"aligned_pairs after needleman_wunsch_align","data":{"aligned_pairs_count":len(aligned_pairs),"aligned_pairs_preview":aligned_pairs[:10],"has_dash_in_pairs":any(exp == '-' or rec == '-' for exp, rec in aligned_pairs),"pairs_with_dash":aligned_pairs_with_dash},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            print(f"Aligned pairs: {len(aligned_pairs)}, score: {alignment_score:.2f}")
            
            # Stage 7: PER Calculation
            per_result = calculate_per(aligned_pairs)
            print(f"PER: {per_result['per']:.2%} (Substitutions: {per_result['substitutions']}, "
                  f"Deletions: {per_result['deletions']}, Insertions: {per_result['insertions']})")
            
            # Stage 8: Diagnostic Analysis
            diagnostic_results = diagnostic_engine.analyze_pronunciation(aligned_pairs)
            
            # Stage 9: Optional Validation
            if enable_validation and optional_validator:
                # For each error, try to validate with trained model
                for i, result in enumerate(diagnostic_results):
                    if not result.get('is_correct', False) and not result.get('is_missing', False):
                        expected_ph = result.get('expected', '')
                        recognized_ph = result.get('recognized', '')
                        
                        if optional_validator.has_trained_model(expected_ph, recognized_ph):
                            # Find corresponding segment
                            segment = None
                            for seg in recognized_segments:
                                if seg.label == recognized_ph:
                                    segment = seg
                                    break
                            
                            if segment:
                                # Extract audio segment
                                start_sample = int(segment.start_time * config.SAMPLE_RATE)
                                end_sample = int(segment.end_time * config.SAMPLE_RATE)
                                audio_segment = waveform[start_sample:end_sample]
                                
                                # Validate
                                validation_result = optional_validator.validate_phoneme_segment(
                                    audio_segment,
                                    phoneme_pair=f"{expected_ph}-{recognized_ph}",
                                    expected_phoneme=expected_ph,
                                    suspected_phoneme=recognized_ph,
                                    sr=config.SAMPLE_RATE
                                )
                                
                                # Update result
                                result['validation_result'] = validation_result
                                if validation_result.get('is_correct'):
                                    result['is_correct'] = True  # Override if validation says correct
            
            # Stage 10: Visualization
            # Output 0: Text with source information (for debugging)
            from modules.visualization import create_text_with_sources_display
            text_with_sources_html = create_text_with_sources_display(
                text_for_phonemes,
                expected_phonemes_dict
            )
            
            # Output 1: Expected phonemes (show expected phonemes directly, with spaces between phonemes)
            expected_phonemes_str = ' '.join(expected_phonemes)
            expected_html = f"<div style='font-family: monospace; font-size: 14px;'><p>{expected_phonemes_str}</p></div>"
            
            # Output 2: Recognized phonemes
            # Use raw_phonemes directly - model already outputs accurate IPA phonemes without filtering
            
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"app.py:456","message":"raw_phonemes before create_simple_phoneme_comparison","data":{"raw_phonemes":raw_phonemes,"count":len(raw_phonemes),"joined":" ".join(raw_phonemes)},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            recognized_html = create_simple_phoneme_comparison([], raw_phonemes)
            
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"app.py:461","message":"recognized_html after create_simple_phoneme_comparison","data":{"html_length":len(recognized_html),"html_preview":recognized_html[:200]},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            # Output 3: Side-by-side comparison
            side_by_side_html = create_side_by_side_comparison(
                expected_phonemes,
                raw_phonemes,
                aligned_pairs
            )
            
            # #region agent log
            import re
            side_by_side_dashes = [m.start() for m in re.finditer(r'[^<>\'"]-', side_by_side_html)]
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"L","location":"app.py:487","message":"side_by_side_html after create_side_by_side_comparison","data":{"html_length":len(side_by_side_html),"html_preview":side_by_side_html[:300],"dash_count":side_by_side_html.count('-'),"dashes_in_text":side_by_side_dashes[:20]},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            # Output 4: Colored text
            # Convert aligned_pairs to dict format for visualization
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
            colored_text_html = create_colored_text(text_for_colored, aligned_pairs_dict)
            
            # Output 5: Detailed report (with WER and PER)
            # Don't show WER if text was empty (WER not calculated)
            show_wer_in_report = config.SHOW_WER and not text_is_empty and wer_result is not None
            text_for_report = recognized_text if text_is_empty else text
            detailed_report_html = create_detailed_report(
                aligned_pairs_dict,
                diagnostic_results,
                text_for_report,
                wer_result=wer_result if show_wer_in_report else None,
                per_result=per_result if config.SHOW_PER else None,
                recognized_text=recognized_text
            )
            
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
                    <li><strong>Optional validation:</strong> {'Enabled' if enable_validation else 'Disabled'}</li>
                </ul>
            </div>
            """
            
            # Output 7: Raw phonemes (before filtering)
            raw_phonemes_html = create_raw_phonemes_display(raw_phonemes)
            
            # Load trimmed audio for return (don't delete it yet)
            trimmed_audio_data = None
            if trimmed_audio_path and Path(trimmed_audio_path).exists():
                try:
                    trimmed_audio, trimmed_sr = sf.read(trimmed_audio_path)
                    # Convert to mono if stereo
                    if len(trimmed_audio.shape) > 1:
                        trimmed_audio = np.mean(trimmed_audio, axis=1)
                    trimmed_audio_data = (trimmed_sr, trimmed_audio)
                except Exception as e:
                    print(f"Warning: Failed to load trimmed audio for output: {e}")
            
            # #region agent log
            import re
            recognized_dashes_in_text = [m.start() for m in re.finditer(r'[^<>\'"]-', recognized_html)]
            side_by_side_dashes_in_text = [m.start() for m in re.finditer(r'[^<>\'"]-', side_by_side_html)]
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"app.py:575","message":"before return from process_pronunciation","data":{"recognized_html_full":recognized_html,"recognized_html_has_dash":'-' in recognized_html,"recognized_html_dash_count":recognized_html.count('-'),"recognized_dashes_in_text":recognized_dashes_in_text,"side_by_side_html_preview":side_by_side_html[:400],"side_by_side_dashes_in_text":side_by_side_dashes_in_text},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            return (
                text_with_sources_html,  # First output: text with sources
                expected_html,
                recognized_html,
                side_by_side_html,
                colored_text_html,
                detailed_report_html,
                technical_html,
                raw_phonemes_html,
                trimmed_audio_data
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
        return (error_html, error_html, error_html, error_html, error_html, error_html, error_html, error_html, None)


def load_models_in_background():
    """Load models in background: first ASR, then dictionaries."""
    print("Starting background model loading...")
    try:
        # Stage 1: Load ASR (Whisper) first - this is critical for user experience
        print("Stage 1: Loading ASR (Whisper)...")
        initialize_asr_only()
        print("ASR loaded successfully!")
        
        # Stage 2: Load dictionaries after ASR is ready
        print("Stage 2: Loading G2P dictionaries...")
        load_dictionaries_in_background()
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
                trimmed_audio_output = gr.Audio(
                    label="8. Processed Audio (VAD disabled)",
                    type="numpy",
                    visible=True
                )
        
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
                raw_phonemes_output,
                trimmed_audio_output
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


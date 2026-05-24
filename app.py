"""
German Pronunciation Diagnostic App (L2-Trainer)
Main application with Gradio interface.
"""

import gradio as gr

# gradio_client ≤1.3 crashes when additionalProperties is a bool in a JSON schema
# (raises TypeError: argument of type 'bool' is not iterable).
# The root route calls get_api_info() which hits this path; url_ok() then sees
# a 500 and raises ValueError: "When localhost is not accessible...".
# Patch before any routes are served.
try:
    import gradio_client.utils as _gcu
    _orig_schema_fn = _gcu._json_schema_to_python_type
    def _patched_schema_fn(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_schema_fn(schema, defs)
    _gcu._json_schema_to_python_type = _patched_schema_fn
except Exception:
    pass

# starlette>=0.41 removed the deprecated TemplateResponse(name, context_dict) signature.
# gradio 4.44.1 still uses the old form; the root route then gets an unhashable-dict
# error which causes url_ok() to see a 500 and raises ValueError about localhost.
# Restore backward compat by converting old-style calls to new-style.
try:
    from starlette.templating import Jinja2Templates as _J2T
    _orig_tr = _J2T.TemplateResponse
    def _compat_tr(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            ctx = dict(args[1]) if len(args) > 1 else {}
            req = ctx.pop("request", None)
            return _orig_tr(self, req, name, ctx, **kwargs)
        return _orig_tr(self, *args, **kwargs)
    _J2T.TemplateResponse = _compat_tr
except Exception:
    pass

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
import tempfile
import sys
import time
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
from modules.utils import collapse_consecutive_duplicates
from modules.chat_utils import normalize_chat_history, add_to_chat_history
from modules.component_manager import (
    initialize_components, load_models_in_background,
    initialize_asr_only, load_dictionaries_in_background,
    load_phoneme_model_in_background, load_mfa_in_background,
    initialize_and_preload_validator,
)
import modules.component_manager as component_manager
from modules.phoneme_validation import validate_single_phoneme
from modules.ui import GRADIO_CUSTOM_CSS


# ---------------------------------------------------------------------------
# Startup loading helpers
# ---------------------------------------------------------------------------

def _startup_html(done: list, current: Optional[str]) -> str:
    """Build the HTML string shown in the startup status component."""
    if current is None:
        # All done: collapse to a single slim bar
        return (
            '<div style="font-family:monospace;font-size:12px;padding:5px 14px;'
            'background:#052e16;border-radius:6px;border:1px solid #16a34a;'
            'color:#22c55e;text-align:center;">'
            '&#10003;&nbsp;All models loaded — application ready to use.'
            '</div>'
        )
    rows = []
    for item in done:
        rows.append(
            f'<div style="color:#22c55e;margin:2px 0;">&#10003;&nbsp;<b>{item}</b></div>'
        )
    rows.append(
        f'<div style="color:#f59e0b;margin:2px 0;">&#8987;&nbsp;{current}</div>'
    )
    inner = "\n".join(rows)
    return (
        '<div style="font-family:monospace;font-size:13px;line-height:1.7;'
        'padding:10px 14px;background:#111827;border-radius:8px;'
        'border:1px solid #374151;">'
        + inner
        + "</div>"
    )


def _startup_loading_sequence():
    """
    Generator executed by app.load().
    Loads every component sequentially and yields HTML status updates.
    Each yield immediately updates the status component in the browser.
    """
    _btn_loading = gr.update(interactive=False)
    _btn_ready   = gr.update(interactive=True)

    # If everything is already loaded (e.g. a second browser tab opens), skip.
    cm = component_manager
    if (
        cm.asr_recognizer is not None
        and cm.phoneme_recognizer is not None
        and cm.optional_validator is not None
        and getattr(cm.optional_validator, "validator", None) is not None
        and len(getattr(cm.optional_validator.validator, "models_cache", {})) > 0
    ):
        yield _startup_html(["All models already loaded"], None), _btn_ready
        return

    done: list[str] = []

    # --- Step 1: ASR ---
    _asr_engine = getattr(__import__('config'), 'ASR_ENGINE', 'whisper')
    if _asr_engine == 'openai':
        _asr_label = f"OpenAI ASR ({getattr(__import__('config'), 'OPENAI_ASR_MODEL', 'gpt-4o-transcribe')})"
    elif _asr_engine == 'groq':
        _asr_label = f"Groq ASR ({getattr(__import__('config'), 'GROQ_MODEL', 'whisper-large-v3')})"
    elif _asr_engine == 'macos':
        _asr_label = "macOS Speech"
    else:
        _asr_label = f"Whisper ASR ({getattr(__import__('config'), 'ASR_MODEL', 'medium')})"
    yield _startup_html(done, f"Loading {_asr_label}..."), _btn_loading
    try:
        initialize_asr_only()
        done.append(_asr_label)
    except Exception as exc:
        done.append(f"{_asr_label} — WARNING: {exc}")

    # --- Step 2: G2P dictionaries + eSpeak warmup ---
    yield _startup_html(done, "Loading G2P dictionaries + warming up eSpeak..."), _btn_loading
    try:
        load_dictionaries_in_background()
        done.append("G2P dictionaries (loaded + warmed up)")
    except Exception as exc:
        done.append(f"G2P dictionaries — WARNING: {exc}")

    # --- Step 3: wav2vec2 phoneme model + warmup ---
    yield _startup_html(done, "Loading wav2vec2 phoneme model + warmup inference (compiling PyTorch ops)..."), _btn_loading
    try:
        load_phoneme_model_in_background()
        done.append("wav2vec2 phoneme model (loaded + warmed up)")
    except Exception as exc:
        done.append(f"wav2vec2 — WARNING: {exc}")

    # --- Step 4: Remaining lightweight components ---
    yield _startup_html(done, "Initializing audio normalizer, aligner, diagnostic engine..."), _btn_loading
    try:
        import modules.component_manager as _cm
        import config as _cfg
        from modules.audio_normalizer import get_audio_normalizer
        from modules.phoneme_filtering import get_phoneme_filter
        from modules.forced_alignment import get_forced_aligner
        from modules.diagnostic_engine import get_diagnostic_engine

        if _cm.audio_normalizer is None:
            _cm.audio_normalizer = get_audio_normalizer()
        if _cm.phoneme_filter is None:
            _cm.phoneme_filter = get_phoneme_filter(
                whitelist=_cfg.PHONEME_WHITELIST,
                confidence_threshold=_cfg.CONFIDENCE_THRESHOLD,
            )
        if _cm.forced_aligner is None:
            _cm.forced_aligner = get_forced_aligner(blank_id=_cfg.FORCED_ALIGNMENT_BLANK_ID)
        if _cm.diagnostic_engine is None:
            _cm.diagnostic_engine = get_diagnostic_engine()
        done.append("Audio normalizer, phoneme filter, forced aligner, diagnostic engine")
    except Exception as exc:
        done.append(f"Support components — WARNING: {exc}")

    # --- Step 5: Validator init + eager model preload (22 models) ---
    yield _startup_html(done, "Initializing phoneme validator..."), _btn_loading
    try:
        # Initialise the wrapper (discovers pairs, no model weights yet)
        initialize_and_preload_validator(progress_callback=None)
        inner = getattr(component_manager.optional_validator, "validator", None)
    except Exception as exc:
        inner = None
        done.append(f"Validator init — WARNING: {exc}")

    if inner is not None:
        pairs = list(getattr(inner, "available_pairs", []))
        total = len(pairs)
        loaded_count = [0]

        for i, pair in enumerate(pairs):
            yield _startup_html(
                done,
                f"Loading validator model {i + 1}/{total}: {pair}...",
            ), _btn_loading
            try:
                if inner._load_model(pair) is not None:
                    loaded_count[0] += 1
            except Exception as exc:
                print(f"Warning: preload failed for '{pair}': {exc}")

        done.append(f"All {loaded_count[0]}/{total} validator models")

    # --- Done: unlock the button ---
    yield _startup_html(done, None), _btn_ready


def process_pronunciation(
    text: str,
    audio_file=None,
    enable_validation: bool = False,
    chat_history=None
):
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
            normalized_audio_path = tmp_path  # Use original audio without normalization
            
            # Stage 1: VAD - Trim noise (DISABLED)
            vad_info = {'enabled': False, 'reason': 'VAD disabled'}
            trimmed_audio_path = normalized_audio_path  # Use normalized audio directly without VAD trimming
            
            # Stage 2: ASR - Speech-to-Text recognition
            asr_start = time.time()
            recognized_text = None
            wer_result = None
            
            # If text is empty, we MUST use ASR to get text from audio
            if text_is_empty:
                if component_manager.asr_recognizer and config.ASR_ENABLED:
                    try:
                        recognized_text = component_manager.asr_recognizer.transcribe(
                            trimmed_audio_path,
                            language=config.ASR_LANGUAGE
                        )
                        asr_elapsed = (time.time() - asr_start) * 1000
                        # Determine ASR engine with detailed logging
                        asr_engine = "unknown"
                        asr_device = "unknown"
                        has_recognizer = hasattr(component_manager.asr_recognizer, 'recognizer')
                        has_model = hasattr(component_manager.asr_recognizer, 'model')
                        asr_type = str(type(component_manager.asr_recognizer))
                        
                        if has_recognizer:
                            asr_engine = "macos"
                            asr_device = "macos_native"  # macOS Speech uses native framework
                        elif has_model:
                            asr_engine = "whisper"
                            # Get device from Whisper recognizer
                            if hasattr(component_manager.asr_recognizer, 'device'):
                                asr_device = component_manager.asr_recognizer.device
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
            elif component_manager.asr_recognizer and config.ASR_ENABLED and config.ASR_ALWAYS_RUN:
                # Text is provided, ASR is optional (for comparison)
                # Only run if ASR_ALWAYS_RUN is enabled to save time
                try:
                    recognized_text = component_manager.asr_recognizer.transcribe(
                        trimmed_audio_path,
                        language=config.ASR_LANGUAGE
                    )
                    asr_elapsed = (time.time() - asr_start) * 1000
                    # Determine ASR engine with detailed logging
                    asr_engine = "unknown"
                    asr_device = "unknown"
                    has_recognizer = hasattr(component_manager.asr_recognizer, 'recognizer')
                    has_model = hasattr(component_manager.asr_recognizer, 'model')
                    asr_type = str(type(component_manager.asr_recognizer))
                    
                    if has_recognizer:
                        asr_engine = "macos"
                        asr_device = "macos_native"  # macOS Speech uses native framework
                    elif has_model:
                        asr_engine = "whisper"
                        # Get device from Whisper recognizer
                        if hasattr(component_manager.asr_recognizer, 'device'):
                            asr_device = component_manager.asr_recognizer.device
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
            logits, emissions = component_manager.phoneme_recognizer.recognize_phonemes(
                trimmed_audio_path,
                sample_rate=config.SAMPLE_RATE
            )
            vocab = component_manager.phoneme_recognizer.get_vocab()
            
            # Decode phonemes (for display)
            decode_start = time.time()
            decoded_phonemes_str = component_manager.phoneme_recognizer.decode_phonemes(logits)
            decode_elapsed = (time.time() - decode_start) * 1000
            phoneme_rec_elapsed = (time.time() - phoneme_rec_start) * 1000            
            raw_phonemes = decoded_phonemes_str.split()
            # Apply CTC collapse logic to recognized phonemes (for consistency, though model already does this)
            raw_phonemes = collapse_consecutive_duplicates(raw_phonemes)            
            print(f"Raw phonemes: {len(raw_phonemes)}")
            
            # Stage 4: Multi-level Filtering
            # Note: Filtering is kept for forced alignment (uses ARPABET), but raw_phonemes are used for display and alignment
            filter_start = time.time()
            filtered_phonemes = component_manager.phoneme_filter.filter_combined(
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
                      component_manager.mfa_aligner is not None and 
                      text and text.strip())
            
            # Choose alignment method: MFA or CTC
            if use_mfa:
                # Use MFA alignment with original text from interface
                try:
                    mfa_align_start = time.time()
                    # Get expected phonemes for MFA (from original text)
                    expected_phonemes_for_mfa = [ph.get('phoneme', '') for ph in get_expected_phonemes(text)]
                    recognized_segments = component_manager.mfa_aligner.extract_phoneme_segments(
                        Path(trimmed_audio_path),
                        text.strip(),
                        expected_phonemes_for_mfa,
                        config.SAMPLE_RATE
                    )
                    mfa_align_elapsed = (time.time() - mfa_align_start) * 1000
                    alignment_method = "MFA"
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
                        recognized_segments = component_manager.forced_aligner.extract_phoneme_segments(
                            waveform_tensor,
                            recognized_phonemes_arpabet,
                            emissions,
                            vocab,
                            config.SAMPLE_RATE
                        )
                        ctc_align_elapsed = (time.time() - ctc_align_start) * 1000
                        alignment_method = "CTC"

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
            diagnostic_results = component_manager.diagnostic_engine.analyze_pronunciation(aligned_pairs)
            diagnostic_elapsed = (time.time() - diagnostic_start) * 1000            
            # Store aligned_pairs before validation for comparison
            aligned_pairs_before_validation = [(exp, rec) for exp, rec in aligned_pairs] if enable_validation else None
            diagnostic_results_before_validation = [dict(dr) for dr in diagnostic_results] if enable_validation else None
            
            # Stage 9: Optional Validation (Parallelized)
            validation_start = time.time()
            validation_count = 0
            validation_corrected_count = 0
            if enable_validation and component_manager.optional_validator:
                # For each mismatch in aligned_pairs, try to validate with trained model
                print(f"Optional validation enabled - checking {len(aligned_pairs)} aligned pairs")
                
                # Step 1: Collect all validation tasks
                # Using validate_phoneme() with full waveform and position_ms (matching notebook implementation)
                validation_tasks = []
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
                    if component_manager.optional_validator.has_trained_model(expected_ph, recognized_ph):
                        # Get proper phoneme pair name
                        phoneme_pair = component_manager.optional_validator.get_phoneme_pair(expected_ph, recognized_ph)
                        if phoneme_pair is None:
                            print(f"Warning: Could not get phoneme pair for {expected_ph} -> {recognized_ph}")
                            segment_index += 1
                            continue
                        
                        # Find corresponding segment using index
                        segment = None
                        if segment_index < len(recognized_segments):
                            segment = recognized_segments[segment_index]
                        
                        if segment:
                            # Calculate position in milliseconds (center of segment)
                            # segment.start_time and segment.end_time are in seconds
                            center_time_seconds = (segment.start_time + segment.end_time) / 2.0
                            position_ms = center_time_seconds * 1000.0
                            
                            # Add task to validation queue
                            # Using validate_phoneme() with full waveform and position_ms (matching notebook)
                            validation_tasks.append({
                                'index': i,
                                'audio': waveform,  # Full waveform, not extracted segment
                                'phoneme': recognized_ph,  # Recognized phoneme (what was detected)
                                'position_ms': position_ms,  # Center of segment in milliseconds
                                'expected_phoneme': expected_ph,  # Expected correct phoneme
                                'phoneme_pair': phoneme_pair,  # For logging/debugging
                                'segment_index': segment_index,
                                'validator': component_manager.optional_validator
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
            if enable_validation and component_manager.optional_validator:
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
                    <li><strong>ASR:</strong> {'Enabled' if (component_manager.asr_recognizer and config.ASR_ENABLED) else 'Disabled'}</li>
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


def create_interface():
    """Create Gradio interface."""

    with gr.Blocks(title="German Pronunciation Diagnostic App (L2-Trainer)", theme=gr.themes.Monochrome(), css=GRADIO_CUSTOM_CSS) as app:
        gr.Markdown("""
        # German Pronunciation Diagnostic App (L2-Trainer)
        """)

        # Startup loading status — visible while models load, shrinks to one line when done.
        startup_status = gr.HTML(
            value=_startup_html([], "Starting up — loading models, please wait..."),
            visible=True,
        )
        app.load(
            fn=_startup_loading_sequence,
            inputs=None,
            outputs=[startup_status, process_btn],
        )
        
        # Main layout: Chatbot on left (70%), controls on right (30%)
        with gr.Row():
            # Left side: Chatbot (70% width)
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Pronunciation Analysis",
                    height=600,
                    show_label=False,
                    container=True,
                    type="messages"
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
                        process_btn = gr.Button("Validate Pronunciation", variant="primary", size="sm", interactive=False)
        
        def apply_gallery_example(sentence: str):
            """Set example sentence and clear any recorded/uploaded audio."""
            return sentence, gr.update(value=None)

        # Examples - first row (run_on_click + fn clears Audio on gallery click)
        gr.Examples(
            examples=[
                ["Hallo, wie geht es dir?"],
                ["Ich habe einen Apfel."],
                ["Der Bär trägt einen Ball."],
                ["Ich möchte ein Stück Kuchen."],
            ],
            inputs=[text_input],
            fn=apply_gallery_example,
            outputs=[text_input, audio_input],
            run_on_click=True,
        )
        
        # Custom button for loading text and audio
        def load_example_text_and_audio():
            """Load example text and audio file."""
            text = "Im Grundlagenstreit der Mathematik entspräche der nominalistischen Position die formalistische Richtung."
            # Use relative path from project root
            audio_path = PROJECT_ROOT / "data" / "audio" / "4aeeae88-0777-2c8c-5c93-2e844a462e49---7c5cf6a7351fb3ca39004d5e49566c09.wav"
            
            # Load audio file if it exists
            if audio_path.exists():
                try:
                    audio_array, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
                    # Gradio expects (sample_rate, audio_array) tuple
                    audio_tuple = (sample_rate, audio_array)
                    return text, audio_tuple
                except Exception as e:
                    print(f"Error loading audio file: {e}")
            
            # Return text only if audio is not available
            return text, None
        
        # Function for second example
        def load_example_text_and_audio_2():
            """Load example text and audio file (example 2)."""
            text = """Aber für unsere Entwicklungspolitik, für unsere Außenpolitik, für unsere Kulturpolitik durch die Goethe-Institute ist das Thema „Teilhabe von Frauen" ein zentrales Thema."""
            # Use relative path from project root
            audio_path = PROJECT_ROOT / "data" / "audio" / "4aeeae88-0777-2c8c-5c93-2e844a462e49---0a05b797c25f88e74d0d8d69a4705187.wav"
            
            # Load audio file if it exists
            if audio_path.exists():
                try:
                    audio_array, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
                    # Gradio expects (sample_rate, audio_array) tuple
                    audio_tuple = (sample_rate, audio_array)
                    return text, audio_tuple
                except Exception as e:
                    print(f"Error loading audio file: {e}")
            
            # Return text only if audio is not available
            return text, None
        
        # Function for third example
        def load_example_text_and_audio_3():
            """Load example text and audio file (example 3)."""
            text = "Plötzlich wurde dem Privatdetektiv klar, worum es dem Dieb eigentlich ging."
            # Use relative path from project root
            audio_path = PROJECT_ROOT / "data" / "audio" / "4aeeae88-0777-2c8c-5c93-2e844a462e49---8da112ef2540faff1fe1dfdf3f433e54.wav"
            
            # Load audio file if it exists
            if audio_path.exists():
                try:
                    audio_array, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
                    # Gradio expects (sample_rate, audio_array) tuple
                    audio_tuple = (sample_rate, audio_array)
                    return text, audio_tuple
                except Exception as e:
                    print(f"Error loading audio file: {e}")
            
            # Return text only if audio is not available
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


def _get_env_bool(name: str, default: bool) -> bool:
    """Read boolean environment variable with sane defaults."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    app = create_interface()

    # Models are now loaded via app.load() inside create_interface(),
    # so no separate background thread is needed here.

    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = _get_env_bool("GRADIO_SHARE", False)
    show_api = _get_env_bool("GRADIO_SHOW_API", False)

    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_api=show_api
    )


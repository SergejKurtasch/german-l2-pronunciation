"""
Configuration file for German Pronunciation Diagnostic App (L2-Trainer).
"""

from pathlib import Path
from typing import List

# Project root
PROJECT_ROOT = Path(__file__).parent

# Model configuration
# Using facebook/wav2vec2-xlsr-53-espeak-cv-ft for phoneme recognition.
# This model is fine-tuned on multi-lingual Common Voice for phoneme recognition.
# It outputs phonetic labels directly and is loaded via Wav2Vec2ForCTC.
MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"

# ASR (Speech-to-Text) settings
# Using OpenAI Whisper for transcribing audio to text
ASR_ENABLED = True  # Enable/disable ASR functionality
ASR_MODEL = "medium"  # Whisper model size: tiny, base, small, medium, large
ASR_LANGUAGE = "de"  # Language code for transcription (German)
ASR_DEVICE = None  # Device for ASR (None = auto-detect, "cpu", "cuda", "mps")

# WER (Word Error Rate) threshold for skipping phoneme analysis
# If WER > WER_THRESHOLD, skip detailed phoneme analysis and show only text comparison
WER_THRESHOLD = 0.70  # 0.70 = 70% error rate (30% words correct)
WER_SKIP_PHONEME_ANALYSIS = True  # Enable/disable skipping phoneme analysis when WER is high

# Metrics display settings
SHOW_WER = True  # Show WER metric in results
SHOW_PER = True  # Show PER (Phoneme Error Rate) metric in results

# Diagnostic Matrix path
DIAGNOSTIC_MATRIX_PATH = PROJECT_ROOT / "diagnostic_matrix.json"

# Confidence threshold for phoneme filtering
# Lower = more phonemes (may include errors), Higher = fewer phonemes (more strict)
CONFIDENCE_THRESHOLD = 0.25  # Balanced threshold (0.3 was too low, 0.5 too high)
CONFIDENCE_THRESHOLD_UNCLEAR = 0.1  # Below this, mark as "unclear"
CONFIDENCE_THRESHOLD_STRICT = 0.6  # For high-confidence filtering (optional)

# Beam search decoding parameters
# Beam search significantly improves accuracy compared to greedy decoding
BEAM_SEARCH_ENABLED = True  # Use beam search instead of greedy decoding
BEAM_WIDTH = 10  # Number of beams for beam search (10-20 recommended, higher = better but slower)
BEAM_SEARCH_LENGTH_PENALTY = 0.5  # Length penalty (0.3-0.6, lower = prefer longer sequences)

# Quick tuning presets (uncomment one to use):
# PRESET_AGGRESSIVE = True  # More strict filtering, fewer errors but may miss phonemes
# PRESET_LENIENT = True  # More lenient filtering, catches more phonemes but may include errors
# PRESET_BALANCED = True  # Balanced (default, no need to uncomment)

# Apply presets if specified
if locals().get('PRESET_AGGRESSIVE', False):
    CONFIDENCE_THRESHOLD = 0.4
    BEAM_WIDTH = 15
    BEAM_SEARCH_LENGTH_PENALTY = 0.4
elif locals().get('PRESET_LENIENT', False):
    CONFIDENCE_THRESHOLD = 0.3
    BEAM_WIDTH = 20
    BEAM_SEARCH_LENGTH_PENALTY = 0.3
# PRESET_BALANCED uses default values above

# Forced Alignment settings
FORCED_ALIGNMENT_BLANK_ID = 0

# Needleman-Wunsch alignment parameters
NW_MATCH_SCORE = 1.0
NW_MISMATCH_SCORE = -1.0
NW_GAP_PENALTY = -1.0

# VAD settings - ULTRA conservative to avoid cutting off quiet speech at the end
VAD_PADDING_MS = 1000  # Very large padding before and after speech (milliseconds) - ultra conservative
VAD_PADDING_END_MS = 1500  # Extra padding at the END to protect last words (milliseconds)
VAD_METHOD = "auto"  # "auto", "silero", "webrtc", "energy"
# Silero VAD parameters - ultra conservative to avoid cutting off quiet speech
VAD_SILERO_THRESHOLD = 0.05  # Ultra low threshold for maximum sensitivity (default: 0.3, was 0.1)
VAD_SILERO_MIN_SILENCE_MS = 100  # Very low minimum silence duration (default: 500) - keeps almost all audio
# Energy-based VAD parameters
VAD_ENERGY_PERCENTILE = 1  # Ultra low percentile - only bottom 1% is considered silence (was 3%)
# End protection settings
VAD_PROTECT_END_SECONDS = 2.0  # Always keep last N seconds if there's any activity (seconds)

# Audio normalization settings (for AGC issues)
ENABLE_AUDIO_NORMALIZATION = True  # Enable audio normalization before VAD
NORMALIZE_COMPRESS_PEAKS = True  # Compress peaks in beginning (first 2-3 phonemes)
NORMALIZE_PEAK_COMPRESSION_RATIO = 0.3  # Compress top 30% of amplitude in beginning
NORMALIZE_PEAK_COMPRESSION_DURATION_MS = 500.0  # Duration from start to apply compression (ms)
NORMALIZE_METHOD = "adaptive"  # "adaptive", "rms", "peak", "none"

# Audio settings
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1  # Mono

# German IPA phonemes whitelist (all German phonemes + common errors)
GERMAN_IPA_PHONEMES = [
    # Vowels
    'a', 'aː', 'ɛ', 'e', 'eː', 'ə', 'ɪ', 'i', 'iː', 'ɔ', 'o', 'oː', 'œ', 'ø', 'øː',
    'ʊ', 'u', 'uː', 'ʏ', 'y', 'yː',
    # Diphthongs
    'aɪ̯', 'aʊ̯', 'ɔʏ̯',
    # Consonants
    'b', 'p', 'd', 't', 'g', 'k', 'kʰ', 'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'ç', 'x', 'h',
    'j', 'l', 'm', 'n', 'ŋ', 'ʁ', 'ɐ',
    # Affricates
    'pf', 'ts', 'tʃ', 'dʒ',
]

# Common error phonemes (phonemes that learners often produce instead of German ones)
COMMON_ERROR_PHONEMES = [
    'u',  # Instead of ʏ
    'ʃ',  # Instead of ç
    'x',  # Instead of h (in some contexts)
    'r',  # Instead of ʁ or ɐ
    'e',  # Instead of ə
    'ɛ',  # Instead of eː
    'ɪ',  # Instead of iː
    'ʊ',  # Instead of uː
    'ɔ',  # Instead of oː
    'a',  # Instead of aː
]

# ARPAbet to IPA mapping (approximate, for English ARPAbet phonemes)
# Note: This is an approximation since ARPAbet is for English, not German
ARPABET_TO_IPA = {
    # Vowels
    'aa': 'a',  # father
    'ae': 'ɛ',  # cat
    'ah': 'a',  # but
    'aw': 'aʊ̯',  # cow
    'ay': 'aɪ̯',  # buy
    'eh': 'ɛ',  # bed
    'er': 'ɐ',  # bird (approximate for German)
    'ey': 'e',  # bait
    'ih': 'ɪ',  # bit
    'iy': 'i',  # beat
    'ow': 'o',  # boat
    'oy': 'ɔʏ̯',  # boy (approximate)
    'uh': 'ʊ',  # book
    'uw': 'u',  # boot
    # Consonants (many are the same)
    'b': 'b',
    'ch': 'tʃ',
    'd': 'd',
    'dh': 'd',  # this (approximate, German doesn't have this)
    'dx': 'd',  # butter (approximate)
    'f': 'f',
    'g': 'g',
    'hh': 'h',
    'h#': '',  # silence, skip
    'jh': 'dʒ',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'p': 'p',
    'r': 'ʁ',  # approximate
    's': 's',
    'sh': 'ʃ',
    'spn': '',  # spoken noise, skip
    't': 't',
    'th': 't',  # think (approximate)
    'v': 'v',
    'w': 'v',  # approximate
    'y': 'j',
    'z': 'z',
    '|': '',  # silence, skip
}

# Combined whitelist (German phonemes + common errors + ARPAbet phonemes for recognition)
ARPABET_PHONEMES = list(ARPABET_TO_IPA.keys())
PHONEME_WHITELIST = list(set(GERMAN_IPA_PHONEMES + COMMON_ERROR_PHONEMES + ARPABET_PHONEMES))


def convert_arpabet_to_ipa(phoneme: str) -> str:
    """
    Convert ARPABET phoneme to IPA.
    
    Args:
        phoneme: ARPABET phoneme string
        
    Returns:
        IPA phoneme string, or original string if not found in mapping
    """
    return ARPABET_TO_IPA.get(phoneme, phoneme)


def convert_phonemes_to_ipa(phonemes: List[str]) -> List[str]:
    """
    Convert list of phonemes (ARPABET or IPA) to IPA format.
    Skips empty strings (silence tokens).
    
    Args:
        phonemes: List of phoneme strings (can be ARPABET or IPA)
        
    Returns:
        List of IPA phoneme strings
    """
    ipa_phonemes = []
    for ph in phonemes:
        ipa_ph = convert_arpabet_to_ipa(ph)
        if ipa_ph:  # Skip empty strings (silence tokens)
            ipa_phonemes.append(ipa_ph)
    return ipa_phonemes


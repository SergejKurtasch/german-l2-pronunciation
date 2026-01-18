"""
Component manager for global component instances.
Manages initialization and lifecycle of ML/Audio processing components.
"""

import time
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import config

# Import component factories
from modules.audio_normalizer import get_audio_normalizer
from modules.g2p_module import get_g2p_converter
from modules.phoneme_recognition import get_phoneme_recognizer
from modules.phoneme_filtering import get_phoneme_filter
from modules.forced_alignment import get_forced_aligner
from modules.diagnostic_engine import get_diagnostic_engine
from modules.phoneme_validator import get_optional_validator
from modules.speech_to_text import get_speech_recognizer
from modules.mfa_alignment import get_mfa_aligner


# Global component instances
audio_normalizer: Optional[object] = None
phoneme_recognizer: Optional[object] = None
phoneme_filter: Optional[object] = None
mfa_aligner: Optional[object] = None
forced_aligner: Optional[object] = None
diagnostic_engine: Optional[object] = None
optional_validator: Optional[object] = None
asr_recognizer: Optional[object] = None


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
                # Determine which engine was actually used
                actual_engine = "whisper"
                if hasattr(asr_recognizer, 'recognizer'):
                    actual_engine = "macos"
                
                if actual_engine == "macos":
                    print("ASR recognizer (macOS Speech) initialized")
                else:
                    model_name = getattr(config, 'ASR_MODEL', 'medium')
                    if requested_engine == "macos" and actual_engine == "whisper":
                        print(f"ASR recognizer (Whisper {model_name}) initialized "
                              f"(macOS Speech not available, using fallback)")
                    else:
                        print(f"ASR recognizer (Whisper {model_name}) initialized")
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
            print(f"Phoneme recognition model loaded successfully! "
                  f"(took {model_load_elapsed/1000:.2f}s)")
        else:
            print("Phoneme recognition model already loaded.")
    except Exception as e:
        print(f"Warning: Phoneme model loading failed: {e}")


def _find_conda_executable() -> Optional[str]:
    """Find conda executable in common locations."""
    conda_cmd = shutil.which("conda")
    if conda_cmd:
        return conda_cmd
    
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
            return str(path)
    
    # Try CONDA_EXE environment variable
    import os
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe
    
    return None


def _check_mfa_availability(conda_env: str, conda_cmd: Optional[str]) -> bool:
    """Check if MFA is available in conda environment."""
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
            return True
    
    if shutil.which("mfa"):
        return True
    
    # Try using conda to check
    if conda_cmd:
        try:
            result = subprocess.run(
                [conda_cmd, "run", "-n", conda_env, "which", "mfa"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return False


def load_mfa_in_background():
    """Load MFA aligner in background."""
    global mfa_aligner
    
    print("Stage 4: Loading MFA aligner...")
    try:
        if mfa_aligner is None:
            conda_env = config.MFA_CONDA_ENV
            conda_cmd = _find_conda_executable()
            mfa_available = _check_mfa_availability(conda_env, conda_cmd)
            
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


def initialize_components():
    """Initialize all global components."""
    global audio_normalizer, phoneme_recognizer, phoneme_filter
    global forced_aligner, diagnostic_engine, optional_validator
    global asr_recognizer, mfa_aligner
    
    init_components_start = time.time()
    
    # Initialize audio normalizer
    if audio_normalizer is None:
        try:
            audio_normalizer = get_audio_normalizer()
            print("Audio normalizer initialized")
        except Exception as e:
            print(f"Warning: Audio normalizer initialization failed: {e}")
            audio_normalizer = None
    
    # Initialize phoneme recognizer (required)
    if phoneme_recognizer is None:
        try:
            phoneme_recognizer = get_phoneme_recognizer(
                model_name=config.MODEL_NAME,
                device=config.MODEL_DEVICE if config.MODEL_DEVICE != "auto" else None
            )
            print(f"Phoneme recognizer (Wav2Vec2 XLSR-53 eSpeak) initialized "
                  f"with model: {phoneme_recognizer.model_name}")
        except Exception as e:
            print(f"Error: Phoneme recognizer initialization failed: {e}")
            raise
    
    # Initialize phoneme filter
    if phoneme_filter is None:
        phoneme_filter = get_phoneme_filter(
            whitelist=config.PHONEME_WHITELIST,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )
        print("Phoneme filter initialized")
    
    # Initialize forced aligner
    if forced_aligner is None:
        forced_aligner = get_forced_aligner(blank_id=config.FORCED_ALIGNMENT_BLANK_ID)
        print("Forced aligner initialized")
    
    # Initialize diagnostic engine
    if diagnostic_engine is None:
        diagnostic_engine = get_diagnostic_engine()
        print("Diagnostic engine initialized")
    
    # Initialize optional validator
    if optional_validator is None:
        optional_validator = get_optional_validator()
        print("Optional validator initialized")
    
    # Preload G2P dictionaries to avoid lazy loading delay
    g2p_converter = get_g2p_converter(load_dicts_immediately=False)
    if not g2p_converter._dicts_loaded:
        print("Preloading G2P dictionaries...")
        g2p_converter._load_dictionaries()
        print("G2P dictionaries preloaded!")
    
    # Initialize ASR if enabled and not already loaded
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


def load_models_in_background():
    """Load models in background for faster startup."""
    # Start ASR initialization (fast)
    initialize_asr_only()
    
    # Start background loading of other components
    import threading
    
    # Load dictionaries in background
    dict_thread = threading.Thread(target=load_dictionaries_in_background, daemon=True)
    dict_thread.start()
    
    # Load phoneme model in background
    model_thread = threading.Thread(target=load_phoneme_model_in_background, daemon=True)
    model_thread.start()
    
    # Load MFA in background if enabled
    if config.MFA_ENABLED:
        mfa_thread = threading.Thread(target=load_mfa_in_background, daemon=True)
        mfa_thread.start()

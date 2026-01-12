"""
MFA (Montreal Forced Aligner) module for extracting precise phoneme segments.
Uses MFA command-line tool to align audio with text and parse TextGrid results.
"""

import subprocess
import shutil
import tempfile
import uuid
import hashlib
import pickle
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import json
import time

# Import PhonemeSegment from forced_alignment module
from modules.forced_alignment import PhonemeSegment

# Try to import textgrid library for parsing TextGrid files
try:
    import textgrid
    HAS_TEXTGRID = True
except ImportError:
    HAS_TEXTGRID = False
    textgrid = None


@dataclass
class MFAConfig:
    """Configuration for MFA aligner."""
    mfa_dict: str = "german_mfa"
    mfa_model: str = "german_mfa"
    mfa_bin_path: Optional[str] = None
    temp_dir: Optional[Path] = None
    conda_env: str = "speechrec"


class MFAAligner:
    """MFA aligner for extracting phoneme segments from audio using Montreal Forced Aligner."""
    
    def __init__(self, config: Optional[MFAConfig] = None):
        """
        Initialize MFA aligner.
        
        Args:
            config: MFA configuration. If None, uses defaults from config.py
        """
        if config is None:
            from config import MFA_DICT, MFA_MODEL, MFA_BIN_PATH, MFA_TEMP_DIR, MFA_CONDA_ENV
            config = MFAConfig(
                mfa_dict=MFA_DICT,
                mfa_model=MFA_MODEL,
                mfa_bin_path=MFA_BIN_PATH,
                temp_dir=MFA_TEMP_DIR,
                conda_env=MFA_CONDA_ENV
            )
        
        self.config = config
        self.mfa_bin = self._find_mfa_binary()
        self.temp_dir = config.temp_dir or Path(tempfile.gettempdir()) / "mfa_align"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "mfa_cache.pkl"
        self.cache = self._load_cache()
        
        # Check dependencies
        if not HAS_TEXTGRID:
            print("Warning: textgrid library not found. Install with: pip install textgrid")
        
        if not self.mfa_bin:
            raise RuntimeError(
                f"MFA binary not found in conda environment '{self.config.conda_env}'. "
                f"Please ensure MFA is installed: conda install -c conda-forge montreal-forced-aligner -n {self.config.conda_env} -y"
            )
        
        print(f"MFA Aligner initialized: {self.mfa_bin}")
        print(f"MFA cache directory: {self.cache_dir}")
    
    def _find_mfa_binary(self) -> Optional[str]:
        """
        Find MFA binary in conda environment or system PATH.
        
        Returns:
            Path to MFA binary or None if not found
        """
        # First, try to use provided path
        if self.config.mfa_bin_path:
            if Path(self.config.mfa_bin_path).exists():
                return str(self.config.mfa_bin_path)
        
        # Find conda executable
        conda_cmd = self._find_conda()
        
        # Try direct paths to MFA in common conda locations
        conda_env = self.config.conda_env
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
                return str(mfa_path)
        
        # Try using conda run to find MFA
        if conda_cmd:
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
                        return mfa_path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Try system PATH
        mfa_path = shutil.which("mfa")
        if mfa_path:
            return mfa_path
        
        return None
    
    def _find_conda(self) -> Optional[str]:
        """Find conda executable in common locations."""
        # Try PATH first
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
    
    def _load_cache(self) -> dict:
        """Load MFA alignment cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"MFA: Loaded cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                print(f"Warning: Failed to load MFA cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save MFA alignment cache to disk."""
        try:
            # Limit cache size to 1000 entries
            if len(self.cache) > 1000:
                items = list(self.cache.items())
                self.cache = dict(items[-1000:])
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Failed to save MFA cache: {e}")
    
    def _get_cache_key(self, audio_path: Path, text: str) -> str:
        """Generate cache key from audio file hash and text."""
        try:
            # Hash first 1MB of audio for speed
            with open(audio_path, 'rb') as f:
                audio_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()
        except Exception:
            # Fallback: use file size and mtime
            stat = audio_path.stat()
            audio_hash = hashlib.md5(f"{stat.st_size}_{stat.st_mtime}".encode()).hexdigest()
        
        text_hash = hashlib.md5(text.strip().encode('utf-8')).hexdigest()
        return f"{audio_hash}_{text_hash}"
    
    def extract_phoneme_segments(
        self,
        audio_path: Path,
        text: str,
        phonemes: Optional[List[str]] = None,
        sample_rate: int = 16000
    ) -> List[PhonemeSegment]:
        """
        Extract phoneme segments using MFA alignment.
        
        Args:
            audio_path: Path to audio file (WAV format, 16kHz recommended)
            text: Text transcription (German text from interface)
            phonemes: Optional list of expected phonemes (for validation)
            sample_rate: Sample rate of audio (default: 16000)
            
        Returns:
            List of PhonemeSegment objects with timing information
        """
        align_start = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(audio_path, text)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            print(f"MFA: Using cached result (saved {cached_result.get('mfa_elapsed_ms', 0):.0f}ms)")
            return cached_result['segments']
        
        # Create unique temporary directory for this alignment
        temp_id = str(uuid.uuid4())[:8]
        temp_corpus = self.temp_dir / f"corpus_{temp_id}"
        temp_output = self.temp_dir / f"output_{temp_id}"
        
        try:
            # Create corpus directory structure
            temp_corpus.mkdir(parents=True, exist_ok=True)
            temp_output.mkdir(parents=True, exist_ok=True)
            
            # Use symlink instead of copy (much faster, like in notebook)
            audio_filename = f"utterance_{temp_id}.wav"
            corpus_audio = temp_corpus / audio_filename
            audio_path_obj = Path(audio_path) if not isinstance(audio_path, Path) else audio_path
            try:
                corpus_audio.symlink_to(audio_path_obj)
            except (OSError, NotImplementedError):
                # Fallback to copy if symlink fails (Windows or cross-filesystem)
                shutil.copy2(audio_path_obj, corpus_audio)
            
            # Create .lab file with text transcription
            lab_file = temp_corpus / f"utterance_{temp_id}.lab"
            with open(lab_file, 'w', encoding='utf-8') as f:
                f.write(text.strip())
            
            # Run MFA alignment
            mfa_start = time.time()
            
            # Use direct MFA binary (much faster than conda run, like in notebook)
            if self.mfa_bin and Path(self.mfa_bin).exists():
                # Direct execution - avoids conda run overhead
                cmd = [
                    self.mfa_bin, "align",
                    str(temp_corpus),
                    self.config.mfa_dict,
                    self.config.mfa_model,
                    str(temp_output),
                    "--clean",
                    "--overwrite",
                    "--num_jobs", "2"  # Use 2 jobs even for single file (helps with model loading)
                ]
                # Set environment to use conda environment's Python and libraries
                env = os.environ.copy()
                conda_env_path = Path(self.mfa_bin).parent.parent
                env_path = str(conda_env_path / "bin")
                if "PATH" in env:
                    env["PATH"] = f"{env_path}:{env['PATH']}"
                else:
                    env["PATH"] = env_path
            else:
                # Fallback to conda run (slower)
                conda_cmd = self._find_conda()
                if not conda_cmd:
                    raise RuntimeError("conda not found. Please ensure conda is installed and in PATH or set CONDA_EXE")
                
                cmd = [
                    conda_cmd, "run", "-n", self.config.conda_env,
                    "mfa", "align",
                    str(temp_corpus),
                    self.config.mfa_dict,
                    self.config.mfa_model,
                    str(temp_output),
                    "--clean",
                    "--overwrite",
                    "--num_jobs", "2"
                ]
                env = None
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )
            
            mfa_elapsed = (time.time() - mfa_start) * 1000
            
            if result.returncode != 0:
                error_msg = f"MFA alignment failed: {result.stderr}"
                print(f"Warning: {error_msg}")
                # Log error
                self._log_error("MFA_ALIGNMENT_FAILED", error_msg, mfa_elapsed)
                return []
            
            # Parse TextGrid result
            textgrid_file = temp_output / f"utterance_{temp_id}.TextGrid"
            if not textgrid_file.exists():
                print(f"Warning: TextGrid file not found: {textgrid_file}")
                return []
            
            # Parse TextGrid and extract phoneme segments
            segments = self._parse_textgrid(textgrid_file, phonemes, sample_rate)
            
            total_elapsed = (time.time() - align_start) * 1000
            
            # Cache result
            self.cache[cache_key] = {
                'segments': segments,
                'timestamp': int(time.time() * 1000),
                'mfa_elapsed_ms': mfa_elapsed
            }
            self._save_cache()
            
            # Log success
            self._log_success(len(segments), mfa_elapsed, total_elapsed, audio_path)
            
            return segments
            
        except subprocess.TimeoutExpired:
            print("Warning: MFA alignment timed out (>60s)")
            self._log_error("MFA_TIMEOUT", "Alignment timed out", 60000)
            return []
        except Exception as e:
            print(f"Warning: MFA alignment error: {e}")
            import traceback
            self._log_error("MFA_EXCEPTION", str(e), 0)
            return []
        finally:
            # Cleanup temporary files
            try:
                if temp_corpus.exists():
                    shutil.rmtree(temp_corpus)
                if temp_output.exists():
                    shutil.rmtree(temp_output)
            except Exception as e:
                print(f"Warning: Failed to cleanup MFA temp files: {e}")
    
    def _parse_textgrid(
        self,
        textgrid_path: Path,
        expected_phonemes: Optional[List[str]] = None,
        sample_rate: int = 16000
    ) -> List[PhonemeSegment]:
        """
        Parse TextGrid file and extract phoneme segments.
        
        Args:
            textgrid_path: Path to TextGrid file
            expected_phonemes: Optional list of expected phonemes for validation
            sample_rate: Sample rate for frame calculation
            
        Returns:
            List of PhonemeSegment objects
        """
        if not HAS_TEXTGRID:
            print("Error: textgrid library not available for parsing")
            return []
        
        try:
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            
            segments = []
            
            # Find phone tier (MFA typically uses "phones" tier)
            phone_tier = None
            for tier in tg.tiers:
                if tier.name.lower() in ["phones", "phone", "phonemes", "phoneme"]:
                    phone_tier = tier
                    break
            
            if not phone_tier:
                # Try first tier if no phone tier found
                if len(tg.tiers) > 0:
                    phone_tier = tg.tiers[0]
                else:
                    print("Warning: No phone tier found in TextGrid")
                    return []
            
            # Extract segments from phone tier
            for interval in phone_tier:
                if interval.mark and interval.mark.strip():  # Skip empty intervals
                    phoneme = interval.mark.strip()
                    
                    # Calculate frame indices
                    # Assuming 20ms per frame (typical for MFA)
                    frame_duration = 0.02  # 20ms
                    start_frame = int(interval.minTime / frame_duration)
                    end_frame = int(interval.maxTime / frame_duration)
                    
                    segments.append(PhonemeSegment(
                        label=phoneme,
                        start_time=interval.minTime,
                        end_time=interval.maxTime,
                        score=1.0,  # MFA doesn't provide confidence scores
                        frame_start=start_frame,
                        frame_end=end_frame
                    ))
            
            return segments
            
        except Exception as e:
            print(f"Error parsing TextGrid: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _log_success(
        self,
        segments_count: int,
        mfa_elapsed_ms: float,
        total_elapsed_ms: float,
        audio_path: Path
    ):
        """Log successful MFA alignment."""
        try:
            log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
            log_entry = {
                "sessionId": "debug-session",
                "runId": "performance",
                "hypothesisId": "MFA_ALIGNMENT_SUCCESS",
                "location": "mfa_alignment.py:extract_phoneme_segments",
                "message": "MFA alignment completed",
                "data": {
                    "segments_count": segments_count,
                    "mfa_elapsed_ms": mfa_elapsed_ms,
                    "total_elapsed_ms": total_elapsed_ms,
                    "audio_file": str(audio_path.name)
                },
                "timestamp": int(time.time() * 1000),
                "elapsed_ms": int(total_elapsed_ms)
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Silently fail logging
    
    def _log_error(self, error_type: str, error_msg: str, elapsed_ms: float):
        """Log MFA alignment error."""
        try:
            log_path = Path(__file__).parent.parent / ".cursor" / "debug.log"
            log_entry = {
                "sessionId": "debug-session",
                "runId": "performance",
                "hypothesisId": "MFA_ALIGNMENT_ERROR",
                "location": "mfa_alignment.py:extract_phoneme_segments",
                "message": f"MFA alignment error: {error_type}",
                "data": {
                    "error_type": error_type,
                    "error_message": error_msg,
                    "elapsed_ms": elapsed_ms
                },
                "timestamp": int(time.time() * 1000),
                "elapsed_ms": int(elapsed_ms)
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Silently fail logging


# Global instance
_mfa_aligner = None


def get_mfa_aligner(config: Optional[MFAConfig] = None) -> Optional[MFAAligner]:
    """
    Get or create global MFA aligner instance.
    
    Args:
        config: Optional MFA configuration
        
    Returns:
        MFAAligner instance or None if initialization failed
    """
    global _mfa_aligner
    if _mfa_aligner is None:
        try:
            _mfa_aligner = MFAAligner(config)
        except Exception as e:
            print(f"Warning: Failed to initialize MFA aligner: {e}")
            _mfa_aligner = None
    return _mfa_aligner

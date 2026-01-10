"""
Voice Activity Detection (VAD) module for detecting speech segments in audio.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import silero-vad (recommended)
try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    HAS_SILERO_VAD = True
except ImportError:
    HAS_SILERO_VAD = False
    torch = None
    load_silero_vad = None
    get_speech_timestamps = None

# Try to import webrtcvad (fallback)
try:
    import webrtcvad
    HAS_WEBRTC_VAD = True
except ImportError:
    HAS_WEBRTC_VAD = False
    webrtcvad = None

# librosa for fallback method
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    librosa = None


class VoiceActivityDetector:
    """Voice Activity Detector for trimming silence and noise from audio."""
    
    def __init__(self, method: str = 'auto'):
        """
        Initialize VAD detector.
        
        Args:
            method: VAD method to use ('silero', 'webrtc', 'energy', or 'auto')
                   'auto' tries silero first, then webrtc, then energy-based
        """
        self.method = method
        self.silero_model = None
        self.webrtc_vad = None
        
        # Initialize Silero VAD if available
        if HAS_SILERO_VAD and (method == 'auto' or method == 'silero'):
            try:
                self.silero_model, utils = load_silero_vad()
                self.method = 'silero'
                print("Using Silero VAD (recommended)")
            except Exception as e:
                print(f"Warning: Failed to load Silero VAD: {e}")
                if method == 'silero':
                    raise
                self.method = None
        
        # Initialize WebRTC VAD if Silero not available
        if self.method is None or self.method == 'auto':
            if HAS_WEBRTC_VAD:
                try:
                    self.webrtc_vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3, 2 is moderate
                    self.method = 'webrtc'
                    print("Using WebRTC VAD")
                except Exception as e:
                    print(f"Warning: Failed to initialize WebRTC VAD: {e}")
                    if method == 'webrtc':
                        raise
                    self.method = None
        
        # Fallback to energy-based method
        if self.method is None or self.method == 'auto':
            if HAS_LIBROSA:
                self.method = 'energy'
                print("Using energy-based VAD (fallback)")
            else:
                raise RuntimeError("No VAD method available. Install silero-vad, webrtcvad, or librosa.")
    
    def detect_speech_segments(
        self,
        audio_path: str,
        sample_rate: int = 16000
    ) -> Tuple[int, int]:
        """
        Detect speech segments in audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Sample rate of audio (for WebRTC VAD, must be 16000)
            
        Returns:
            Tuple of (start_sample, end_sample) for speech segment
            If no speech found, returns (0, total_samples)
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed (for WebRTC VAD)
        if self.method == 'webrtc' and sr != 16000:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                raise ValueError("WebRTC VAD requires 16kHz audio. Install librosa for resampling.")
        
        total_samples = len(audio)
        
        # Check if audio is too short
        if total_samples < int(sr * 0.1):  # Less than 100ms
            print("Warning: Audio too short (< 100ms), returning full audio")
            return (0, total_samples)
        
        # Detect speech using selected method
        if self.method == 'silero':
            return self._detect_silero(audio, sr)
        elif self.method == 'webrtc':
            return self._detect_webrtc(audio, sr)
        elif self.method == 'energy':
            return self._detect_energy(audio, sr)
        else:
            raise RuntimeError(f"Unknown VAD method: {self.method}")
    
    def _detect_silero(self, audio: np.ndarray, sample_rate: int) -> Tuple[int, int]:
        """Detect speech using Silero VAD."""
        # Import config for VAD parameters
        try:
            import config
            threshold = getattr(config, 'VAD_SILERO_THRESHOLD', 0.2)
            min_silence_ms = getattr(config, 'VAD_SILERO_MIN_SILENCE_MS', 250)
        except ImportError:
            threshold = 0.2
            min_silence_ms = 250
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps with less strict parameters
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.silero_model,
            sampling_rate=sample_rate,
            threshold=threshold,  # Lower threshold for more sensitive detection
            min_speech_duration_ms=100,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=min_silence_ms,  # Lower value to keep more audio between segments
            window_size_samples=512,
            speech_pad_ms=100
        )
        
        if not speech_timestamps:
            print("Warning: No speech detected by Silero VAD, returning full audio")
            return (0, len(audio))
        
        # Merge all speech segments - use first start and last end
        # This ensures we capture all speech, including parts between pauses
        # With lower min_silence_duration_ms, segments are less likely to be split,
        # but if they are, we still capture everything from first to last segment
        start_sample = speech_timestamps[0]['start']
        end_sample = speech_timestamps[-1]['end']
        
        # ULTRA CONSERVATIVE END PROTECTION:
        # If the last detected segment is within the last N seconds, extend to the actual end
        # This protects quiet words at the end
        try:
            import config
            protect_end_seconds = getattr(config, 'VAD_PROTECT_END_SECONDS', 0.0)
        except ImportError:
            protect_end_seconds = 0.0
        
        if protect_end_seconds > 0:
            total_samples = len(audio)
            protect_end_samples = int(sample_rate * protect_end_seconds)
            end_threshold = total_samples - protect_end_samples
            
            # If detected end is in the last N seconds, extend to actual end
            if end_sample >= end_threshold:
                end_sample = total_samples
                print(f"Silero VAD: End protection - extending to actual end (detected at {end_sample/sample_rate:.2f}s)")
        
        return (int(start_sample), int(end_sample))
    
    def _detect_webrtc(self, audio: np.ndarray, sample_rate: int) -> Tuple[int, int]:
        """Detect speech using WebRTC VAD."""
        if sample_rate != 16000:
            raise ValueError("WebRTC VAD requires 16kHz audio")
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32768).astype(np.int16)
        
        # WebRTC VAD works on 10ms, 20ms, or 30ms frames
        frame_duration_ms = 30
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        speech_frames = []
        
        # Process audio in frames
        for i in range(0, len(audio_int16), frame_size):
            frame = audio_int16[i:i + frame_size]
            
            # Pad frame if needed
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
            
            # Check if frame contains speech
            is_speech = self.webrtc_vad.is_speech(frame.tobytes(), sample_rate)
            speech_frames.append((i, is_speech))
        
        # Find first and last speech frames
        speech_indices = [i for i, is_speech in speech_frames if is_speech]
        
        if not speech_indices:
            print("Warning: No speech detected by WebRTC VAD, returning full audio")
            return (0, len(audio))
        
        start_frame = speech_indices[0]
        end_frame = speech_indices[-1]
        
        # Convert frame indices to sample indices
        start_sample = start_frame
        end_sample = min(end_frame + frame_size, len(audio))
        
        # ULTRA CONSERVATIVE END PROTECTION for WebRTC method
        try:
            import config
            protect_end_seconds = getattr(config, 'VAD_PROTECT_END_SECONDS', 0.0)
        except ImportError:
            protect_end_seconds = 0.0
        
        if protect_end_seconds > 0:
            total_samples = len(audio)
            protect_end_samples = int(sample_rate * protect_end_seconds)
            end_threshold = total_samples - protect_end_samples
            
            # If detected end is in the last N seconds, extend to actual end
            if end_sample >= end_threshold:
                end_sample = total_samples
                print(f"WebRTC VAD: End protection - extending to actual end")
        
        return (start_sample, end_sample)
    
    def _detect_energy(self, audio: np.ndarray, sample_rate: int) -> Tuple[int, int]:
        """Detect speech using energy-based method (fallback)."""
        # Calculate frame energy
        frame_length = int(sample_rate * 0.025)  # 25ms frames
        hop_length = int(sample_rate * 0.010)  # 10ms hop
        
        # Calculate RMS energy per frame
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sqrt(np.mean(frames ** 2, axis=0))
        
        # Normalize energy
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
        
        # Threshold for speech detection (adaptive) - very low percentile for very conservative trimming
        try:
            import config
            percentile = getattr(config, 'VAD_ENERGY_PERCENTILE', 5)
        except ImportError:
            percentile = 5
        threshold = np.percentile(energy_norm, percentile)  # Bottom percentile is likely silence
        
        # Find speech frames
        speech_frames = energy_norm > threshold
        
        if not np.any(speech_frames):
            print("Warning: No speech detected by energy-based VAD, returning full audio")
            return (0, len(audio))
        
        # Find first and last speech frame
        speech_indices = np.where(speech_frames)[0]
        start_frame = speech_indices[0]
        end_frame = speech_indices[-1]
        
        # Convert frame indices to sample indices
        start_sample = start_frame * hop_length
        end_sample = min((end_frame + 1) * hop_length, len(audio))
        
        # ULTRA CONSERVATIVE END PROTECTION for energy-based method
        try:
            import config
            protect_end_seconds = getattr(config, 'VAD_PROTECT_END_SECONDS', 0.0)
        except ImportError:
            protect_end_seconds = 0.0
        
        if protect_end_seconds > 0:
            total_samples = len(audio)
            protect_end_samples = int(sample_rate * protect_end_seconds)
            end_threshold = total_samples - protect_end_samples
            
            # If detected end is in the last N seconds, extend to actual end
            if end_sample >= end_threshold:
                end_sample = total_samples
                print(f"Energy VAD: End protection - extending to actual end")
        
        return (int(start_sample), int(end_sample))
    
    def trim_audio(
        self,
        audio_path: str,
        output_path: str,
        sample_rate: int = 16000,
        padding_ms: int = 75
    ) -> str:
        """
        Trim audio to speech segments and save to output file.
        Ultra conservative, especially at the end.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save trimmed audio
            sample_rate: Sample rate of audio
            padding_ms: Padding in milliseconds to add before and after speech
            
        Returns:
            Path to trimmed audio file
        """
        # Import config for end protection
        try:
            import config
            padding_end_ms = getattr(config, 'VAD_PADDING_END_MS', padding_ms)
            protect_end_seconds = getattr(config, 'VAD_PROTECT_END_SECONDS', 0.0)
        except ImportError:
            padding_end_ms = padding_ms
            protect_end_seconds = 0.0
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        total_samples = len(audio)
        total_duration = total_samples / sr
        
        # Detect speech segments
        start_sample, end_sample = self.detect_speech_segments(audio_path, sr)
        
        # Add padding at the beginning
        padding_start_samples = int(sr * padding_ms / 1000)
        start_sample = max(0, start_sample - padding_start_samples)
        
        # Add EXTRA padding at the end (more conservative)
        padding_end_samples = int(sr * padding_end_ms / 1000)
        end_sample = min(total_samples, end_sample + padding_end_samples)
        
        # PROTECT END: Multiple strategies to protect the end
        
        # Strategy 1: If detected end is within protect_end_seconds from actual end,
        # keep everything to the actual end (don't trim the last N seconds)
        if protect_end_seconds > 0:
            protect_end_samples = int(sr * protect_end_seconds)
            end_threshold = total_samples - protect_end_samples
            
            # If detected end is close to the actual end (within protect_end_seconds),
            # keep everything to the actual end
            if end_sample >= end_threshold:
                # There's activity in the last N seconds, keep everything
                end_sample = total_samples
                print(f"End protection (strategy 1): Keeping last {protect_end_seconds}s (detected end at {end_sample/sr:.2f}s, total {total_duration:.2f}s)")
            else:
                # Strategy 2: Check if there's ANY activity in the last N seconds
                # Even if it's below the speech threshold, keep it
                last_segment = audio[end_threshold:]
                if len(last_segment) > 0:
                    # Calculate RMS energy of last segment
                    last_rms = np.sqrt(np.mean(last_segment ** 2))
                    # Calculate RMS energy of entire audio for comparison
                    total_rms = np.sqrt(np.mean(audio ** 2))
                    
                    # If last segment has at least 10% of average energy, keep it
                    # This catches quiet but real speech
                    if last_rms > total_rms * 0.1:
                        end_sample = total_samples
                        print(f"End protection (strategy 2): Last {protect_end_seconds}s has activity (RMS: {last_rms:.4f} vs avg {total_rms:.4f}), keeping to end")
        
        # Additional safety: Never trim more than 95% of the original audio
        # This prevents over-aggressive trimming
        min_keep_samples = int(total_samples * 0.05)  # Keep at least 5%
        if (end_sample - start_sample) < min_keep_samples:
            # If detected segment is too short, expand it
            center = (start_sample + end_sample) // 2
            start_sample = max(0, center - min_keep_samples // 2)
            end_sample = min(total_samples, center + min_keep_samples // 2)
            print(f"Safety: Expanded segment to minimum {min_keep_samples/sr:.2f}s")
        
        # Trim audio
        trimmed_audio = audio[start_sample:end_sample]
        
        # Save trimmed audio
        sf.write(output_path, trimmed_audio, sr)
        
        print(f"Trimmed audio: {len(audio)/sr:.2f}s -> {len(trimmed_audio)/sr:.2f}s (kept {len(trimmed_audio)/len(audio)*100:.1f}%)")
        print(f"  Start: {start_sample/sr:.2f}s, End: {end_sample/sr:.2f}s (original end: {total_duration:.2f}s)")
        
        return output_path


# Global instance
_vad_detector = None


def get_vad_detector(method: str = 'auto') -> VoiceActivityDetector:
    """Get or create global VAD detector instance."""
    global _vad_detector
    if _vad_detector is None:
        _vad_detector = VoiceActivityDetector(method=method)
    return _vad_detector


"""
Audio normalization module for handling AGC (Automatic Gain Control) issues.
Compresses peaks in the beginning of audio and normalizes the entire signal.
"""

import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import librosa for advanced processing
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    librosa = None


class AudioNormalizer:
    """Normalizes audio to handle AGC issues where beginning is loud and end is quiet."""
    
    def __init__(self):
        """Initialize audio normalizer."""
        pass
    
    def normalize_agc_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        compress_peaks: bool = True,
        peak_compression_ratio: float = 0.3,
        peak_compression_duration_ms: float = 500.0,
        normalize_method: str = 'adaptive'
    ) -> np.ndarray:
        """
        Normalize audio to handle AGC issues.
        
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of audio
            compress_peaks: Whether to compress peaks in the beginning
            peak_compression_ratio: Ratio of top amplitude to compress (0.3 = top 30%)
            peak_compression_duration_ms: Duration in ms from start to apply compression
            normalize_method: Normalization method ('adaptive', 'rms', 'peak', 'none')
            
        Returns:
            Normalized audio array
        """
        audio = audio.copy()
        
        # Step 1: Compress peaks in the beginning (first 2-3 phonemes)
        if compress_peaks:
            audio = self._compress_beginning_peaks(
                audio,
                sample_rate,
                compression_ratio=peak_compression_ratio,
                duration_ms=peak_compression_duration_ms
            )
        
        # Step 2: Normalize the entire signal
        if normalize_method != 'none':
            audio = self._normalize_audio(audio, method=normalize_method, sample_rate=sample_rate)
        
        return audio
    
    def _compress_beginning_peaks(
        self,
        audio: np.ndarray,
        sample_rate: int,
        compression_ratio: float = 0.3,
        duration_ms: float = 500.0
    ) -> np.ndarray:
        """
        Compress peaks in the beginning of audio.
        Compresses the top compression_ratio (e.g., 30%) of amplitude values.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            compression_ratio: Ratio of top amplitude to compress (0.3 = top 30%)
            duration_ms: Duration from start to apply compression
            
        Returns:
            Audio with compressed peaks
        """
        audio = audio.copy()
        
        # Calculate number of samples to process
        num_samples = int(sample_rate * duration_ms / 1000.0)
        num_samples = min(num_samples, len(audio))
        
        if num_samples == 0:
            return audio
        
        # Get the beginning segment
        beginning_segment = audio[:num_samples]
        
        # Calculate threshold for top compression_ratio of amplitude
        abs_amplitudes = np.abs(beginning_segment)
        threshold = np.percentile(abs_amplitudes, (1 - compression_ratio) * 100)
        
        # Find samples above threshold
        mask = np.abs(beginning_segment) > threshold
        
        if np.any(mask):
            # Compress: reduce amplitude of peaks
            # Use soft compression: reduce by factor, but not completely
            compression_factor = 0.5  # Reduce peaks by 50%
            
            # Apply compression with smooth transition
            compressed_segment = beginning_segment.copy()
            peak_samples = beginning_segment[mask]
            
            # Soft compression: reduce amplitude above threshold
            excess = np.abs(peak_samples) - threshold
            new_amplitude = threshold + excess * compression_factor
            compressed_segment[mask] = np.sign(peak_samples) * new_amplitude
            
            # Replace beginning segment
            audio[:num_samples] = compressed_segment
        
        return audio
    
    def _normalize_audio(
        self,
        audio: np.ndarray,
        method: str = 'adaptive',
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Normalize audio using different methods.
        
        Args:
            audio: Audio signal
            method: 'adaptive' (time-based), 'rms', 'peak', 'none'
            sample_rate: Sample rate for adaptive normalization
            
        Returns:
            Normalized audio
        """
        if method == 'none':
            return audio
        
        if len(audio) == 0:
            return audio
        
        if method == 'adaptive':
            # Adaptive normalization: normalize based on RMS energy over time
            # This helps when beginning is loud and end is quiet
            return self._adaptive_normalize(audio, sample_rate=sample_rate)
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 1e-10:
                target_rms = 0.1  # Target RMS level
                audio = audio * (target_rms / rms)
            return audio
        elif method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 1e-10:
                audio = audio / max_val * 0.95  # Leave some headroom
            return audio
        else:
            return audio
    
    def _adaptive_normalize(self, audio: np.ndarray, window_size_ms: float = 100.0, sample_rate: int = 16000) -> np.ndarray:
        """
        Adaptive normalization that adjusts gain over time.
        Helps when beginning is loud and end is quiet.
        Uses time-varying gain to boost quiet parts and reduce loud parts.
        
        Args:
            audio: Audio signal
            window_size_ms: Window size for RMS calculation
            sample_rate: Sample rate of audio
            
        Returns:
            Adaptively normalized audio
        """
        if len(audio) == 0:
            return audio
        
        # Calculate RMS in windows
        window_samples = int(sample_rate * window_size_ms / 1000.0)
        window_samples = max(window_samples, 100)  # Minimum window size
        
        # Calculate RMS for each window
        num_windows = len(audio) // window_samples
        if num_windows == 0:
            # Audio too short, use simple RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 1e-10:
                return audio * (0.1 / rms)
            return audio
        
        # Calculate RMS for each window
        window_rms = []
        for i in range(num_windows):
            start = i * window_samples
            end = min(start + window_samples, len(audio))
            window = audio[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            window_rms.append(rms)
        
        # Use median RMS as target (more robust than mean)
        target_rms = np.median(window_rms)
        if target_rms < 1e-10:
            target_rms = 0.1  # Fallback
        
        # Method 1: Simple global normalization to target RMS
        # This helps when beginning is loud and end is quiet
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 1e-10:
            # Normalize to target RMS, but don't over-amplify
            gain = min(target_rms / current_rms, 3.0)  # Max 3x amplification
            audio = audio * gain
        
        return audio
    
    def process_audio_file(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int = 16000,
        compress_peaks: bool = True,
        peak_compression_ratio: float = 0.3,
        peak_compression_duration_ms: float = 500.0,
        normalize_method: str = 'adaptive'
    ) -> str:
        """
        Process audio file: normalize and save.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            sample_rate: Sample rate
            compress_peaks: Whether to compress peaks
            peak_compression_ratio: Ratio of top amplitude to compress
            peak_compression_duration_ms: Duration from start to compress
            normalize_method: Normalization method
            
        Returns:
            Path to processed audio file
        """
        # Load audio
        audio, sr = sf.read(input_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != sample_rate:
            if HAS_LIBROSA:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            else:
                raise ValueError(f"Audio sample rate {sr} != {sample_rate}. Install librosa for resampling.")
        
        # Normalize
        normalized_audio = self.normalize_agc_audio(
            audio,
            sample_rate,
            compress_peaks=compress_peaks,
            peak_compression_ratio=peak_compression_ratio,
            peak_compression_duration_ms=peak_compression_duration_ms,
            normalize_method=normalize_method
        )
        
        # Save
        sf.write(output_path, normalized_audio, sample_rate)
        
        return output_path


# Global instance
_audio_normalizer = None


def get_audio_normalizer() -> AudioNormalizer:
    """Get or create global audio normalizer instance."""
    global _audio_normalizer
    if _audio_normalizer is None:
        _audio_normalizer = AudioNormalizer()
    return _audio_normalizer


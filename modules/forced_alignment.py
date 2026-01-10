"""
Forced Alignment module using torchaudio for extracting precise phoneme segments.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

# Try to import torchaudio forced_align
try:
    import torchaudio.functional as F
    HAS_FORCED_ALIGN = hasattr(F, 'forced_align')
except (ImportError, AttributeError):
    HAS_FORCED_ALIGN = False
    F = None


@dataclass
class PhonemeSegment:
    """Phoneme segment with timing information."""
    label: str
    start_time: float
    end_time: float
    score: float
    frame_start: int
    frame_end: int


class ForcedAligner:
    """Forced aligner for extracting phoneme segments from audio."""
    
    def __init__(self, blank_id: int = 0):
        """
        Initialize forced aligner.
        
        Args:
            blank_id: ID of blank token in CTC (usually 0)
        """
        self.blank_id = blank_id
    
    def extract_phoneme_segments(
        self,
        waveform: torch.Tensor,
        labels: List[str],
        emissions: torch.Tensor,
        dictionary: Dict[str, int],
        sample_rate: int
    ) -> List[PhonemeSegment]:
        """
        Extract phoneme segments using forced alignment.
        
        Args:
            waveform: Audio waveform tensor (1, N)
            labels: List of phoneme labels (expected sequence)
            emissions: Log-softmax emissions from model (batch, time, vocab_size)
            dictionary: Vocabulary mapping (token -> ID)
            sample_rate: Sample rate of audio
            
        Returns:
            List of PhonemeSegment objects with timing information
        """
        # Prepare tokenized labels
        token_ids = []
        for label in labels:
            if label in dictionary:
                token_ids.append(dictionary[label])
            else:
                # Skip unknown tokens
                print(f"Warning: Unknown token '{label}' in dictionary, skipping")
                continue
        
        if not token_ids:
            return []
        
        # Convert to tensor
        tokenized_labels = torch.tensor([token_ids], dtype=torch.int32)
        targets = tokenized_labels.to(waveform.device)
        
        # Perform forced alignment
        if not HAS_FORCED_ALIGN:
            print("Warning: torchaudio.forced_align not available. Using fallback method.")
            return self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
        
        try:
            alignment, scores = F.forced_align(
                emissions,
                targets,
                blank_id=self.blank_id
            )
        except Exception as e:
            print(f"Error in forced alignment: {e}, using fallback")
            return self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
        
        # Calculate stride (time per frame)
        # Wav2Vec2 typically has downsampling factor of 320
        # So each frame corresponds to 320 samples at 16kHz = 20ms
        num_frames = emissions.shape[1]
        num_samples = waveform.shape[1]
        frame_duration = num_samples / num_frames / sample_rate  # seconds per frame
        
        # Extract segments from alignment
        segments = []
        
        # Group consecutive frames with same label
        if len(alignment) == 0:
            return []
        
        current_label_idx = 0
        current_start_frame = alignment[0]
        
        for i in range(1, len(alignment)):
            frame_idx = alignment[i]
            label_idx = targets[0, current_label_idx].item() if current_label_idx < len(token_ids) else -1
            
            # Check if we should start a new segment
            if frame_idx != alignment[i-1] + 1 or current_label_idx >= len(token_ids):
                # End current segment
                if current_label_idx < len(token_ids):
                    label = labels[current_label_idx]
                    start_time = current_start_frame * frame_duration
                    end_time = alignment[i-1] * frame_duration
                    score = scores[current_label_idx].item() if current_label_idx < len(scores) else 0.0
                    
                    segments.append(PhonemeSegment(
                        label=label,
                        start_time=start_time,
                        end_time=end_time,
                        score=score,
                        frame_start=int(current_start_frame),
                        frame_end=int(alignment[i-1])
                    ))
                
                # Start new segment
                current_start_frame = frame_idx
                current_label_idx += 1
                if current_label_idx >= len(token_ids):
                    break
        
        # Add last segment
        if current_label_idx < len(token_ids):
            label = labels[current_label_idx]
            start_time = current_start_frame * frame_duration
            end_time = alignment[-1] * frame_duration
            score = scores[current_label_idx].item() if current_label_idx < len(scores) else 0.0
            
            segments.append(PhonemeSegment(
                label=label,
                start_time=start_time,
                end_time=end_time,
                score=score,
                frame_start=int(current_start_frame),
                frame_end=int(alignment[-1])
            ))
        
        return segments
    
    def _fallback_alignment(
        self,
        labels: List[str],
        emissions: torch.Tensor,
        dictionary: Dict[str, int],
        sample_rate: int,
        waveform: torch.Tensor
    ) -> List[PhonemeSegment]:
        """
        Fallback alignment method when torchaudio.forced_align is not available.
        Uses simple time-based distribution.
        """
        num_frames = emissions.shape[1]
        num_samples = waveform.shape[1]
        frame_duration = num_samples / num_frames / sample_rate
        
        segments = []
        duration_per_phoneme = (num_samples / sample_rate) / len(labels) if labels else 0
        
        for i, label in enumerate(labels):
            start_time = i * duration_per_phoneme
            end_time = (i + 1) * duration_per_phoneme
            
            segments.append(PhonemeSegment(
                label=label,
                start_time=start_time,
                end_time=end_time,
                score=0.5,  # Default score
                frame_start=int(start_time / frame_duration),
                frame_end=int(end_time / frame_duration)
            ))
        
        return segments


# Global instance
_forced_aligner = None


def get_forced_aligner(blank_id: int = 0) -> ForcedAligner:
    """Get or create global forced aligner instance."""
    global _forced_aligner
    if _forced_aligner is None:
        _forced_aligner = ForcedAligner(blank_id=blank_id)
    return _forced_aligner


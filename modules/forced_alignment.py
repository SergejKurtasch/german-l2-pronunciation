"""
Forced Alignment module using torchaudio for extracting precise phoneme segments.
"""

import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

# Try to import torchaudio forced_align
import json, time
HAS_FORCED_ALIGN = False
F = None
TORCHAUDIO_VERSION = None
FORCED_ALIGN_CHECK_RESULT = None

# Try to import CTC Forced Aligner as alternative
HAS_CTC_ALIGNER = False
ctc_aligner = None
try:
    from ctc_forced_aligner import get_alignments, get_spans
    HAS_CTC_ALIGNER = True
    print("CTC Forced Aligner loaded successfully")
except ImportError:
    print("CTC Forced Aligner not available, will use fallback method")
    HAS_CTC_ALIGNER = False

try:
    import torchaudio
    import torchaudio.functional as F
    TORCHAUDIO_VERSION = getattr(torchaudio, '__version__', 'unknown')
    HAS_FORCED_ALIGN = hasattr(F, 'forced_align')
    
    # #region agent log
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"A","location":"forced_alignment.py:module_init","message":"TorchAudio import check","data":{"torchaudio_version":TORCHAUDIO_VERSION,"has_forced_align_attr":HAS_FORCED_ALIGN,"forced_align_type":str(type(getattr(F, 'forced_align', None))) if F else "F is None"},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion
    
    # Try to test if forced_align is actually callable and works
    if HAS_FORCED_ALIGN:
        try:
            # Test with minimal dummy data to check if extension is compiled
            test_emissions = torch.zeros(1, 10, 5)  # (batch, time, vocab)
            test_targets = torch.tensor([[0, 1]], dtype=torch.int32)
            test_result = F.forced_align(test_emissions, test_targets, blank_id=0)
            FORCED_ALIGN_CHECK_RESULT = "works"
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"A","location":"forced_alignment.py:module_init","message":"Forced align test call succeeded","data":{"test_result_type":str(type(test_result))},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
        except Exception as test_e:
            FORCED_ALIGN_CHECK_RESULT = f"test_failed: {str(test_e)}"
            HAS_FORCED_ALIGN = False  # Mark as unavailable if test fails
            # Only print warning once at module import, not during runtime
            error_msg = str(test_e)
            if "alignment extension" in error_msg.lower() or "not compiled" in error_msg.lower():
                # This is expected for most TorchAudio installations
                pass  # Silent - will use fallback method
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"A","location":"forced_alignment.py:module_init","message":"Forced align test call failed","data":{"error_type":type(test_e).__name__,"error_message":str(test_e)},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
except (ImportError, AttributeError) as e:
    FORCED_ALIGN_CHECK_RESULT = f"import_error: {str(e)}"
    # #region agent log
    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"B","location":"forced_alignment.py:module_init","message":"TorchAudio import failed","data":{"error_type":type(e).__name__,"error_message":str(e)},"timestamp":int(time.time()*1000)})+'\n')
    # #endregion


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
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"C","location":"forced_alignment.py:__init__","message":"ForcedAligner initialized","data":{"blank_id":blank_id,"has_forced_align":HAS_FORCED_ALIGN,"has_ctc_aligner":HAS_CTC_ALIGNER,"torchaudio_version":TORCHAUDIO_VERSION,"check_result":FORCED_ALIGN_CHECK_RESULT},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
    
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
        align_start = time.time()
        
        # Prepare tokenized labels
        tokenize_start = time.time()
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
        tokenize_elapsed = (time.time() - tokenize_start) * 1000
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:after_tokenize","message":"Labels tokenized","data":{"labels_count":len(labels),"token_ids_count":len(token_ids)},"timestamp":int(time.time()*1000),"elapsed_ms":int(tokenize_elapsed)})+'\n')
        # #endregion
        
        # Perform forced alignment
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"D","location":"forced_alignment.py:extract_phoneme_segments:before_alignment","message":"Before forced alignment attempt","data":{"has_forced_align":HAS_FORCED_ALIGN,"emissions_shape":list(emissions.shape) if emissions is not None else None,"targets_shape":list(targets.shape) if targets is not None else None,"blank_id":self.blank_id,"torchaudio_version":TORCHAUDIO_VERSION,"check_result":FORCED_ALIGN_CHECK_RESULT},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        if not HAS_FORCED_ALIGN:
            # Try CTC Forced Aligner as alternative
            if HAS_CTC_ALIGNER:
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"CTC","location":"forced_alignment.py:extract_phoneme_segments:using_ctc_aligner","message":"Using CTC Forced Aligner","data":{"check_result":FORCED_ALIGN_CHECK_RESULT},"timestamp":int(time.time()*1000)})+'\n')
                # #endregion
                ctc_start = time.time()
                result = self._ctc_alignment(labels, emissions, dictionary, sample_rate, waveform, token_ids)
                ctc_elapsed = (time.time() - ctc_start) * 1000
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:ctc_alignment","message":"CTC alignment completed","data":{"segments_count":len(result)},"timestamp":int(time.time()*1000),"elapsed_ms":int(ctc_elapsed)})+'\n')
                # #endregion
                return result
            else:
                # Use simple fallback only if CTC aligner is also not available
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"A","location":"forced_alignment.py:extract_phoneme_segments:has_forced_align_false","message":"HAS_FORCED_ALIGN is False, using fallback","data":{"check_result":FORCED_ALIGN_CHECK_RESULT},"timestamp":int(time.time()*1000)})+'\n')
                # #endregion
                fallback_start = time.time()
                result = self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
                fallback_elapsed = (time.time() - fallback_start) * 1000
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:fallback","message":"Fallback alignment completed","data":{"segments_count":len(result)},"timestamp":int(time.time()*1000),"elapsed_ms":int(fallback_elapsed)})+'\n')
                # #endregion
                return result
        
        try:
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"C","location":"forced_alignment.py:extract_phoneme_segments:before_forced_align_call","message":"Attempting forced_align call","data":{"emissions_device":str(emissions.device),"targets_device":str(targets.device),"emissions_dtype":str(emissions.dtype),"targets_dtype":str(targets.dtype)},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
            forced_align_start = time.time()
            alignment, scores = F.forced_align(
                emissions,
                targets,
                blank_id=self.blank_id
            )
            forced_align_elapsed = (time.time() - forced_align_start) * 1000
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"C","location":"forced_alignment.py:extract_phoneme_segments:forced_align_success","message":"Forced align call succeeded","data":{"alignment_length":len(alignment) if alignment is not None else 0,"alignment_type":str(type(alignment)),"scores_type":str(type(scores))},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:after_forced_align","message":"Forced alignment computation completed","data":{"alignment_length":len(alignment) if alignment is not None else 0},"timestamp":int(time.time()*1000),"elapsed_ms":int(forced_align_elapsed)})+'\n')
            # #endregion
        except Exception as e:
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"debug","hypothesisId":"A","location":"forced_alignment.py:extract_phoneme_segments:forced_align_exception","message":"Forced align call raised exception","data":{"error_type":type(e).__name__,"error_message":str(e),"error_repr":repr(e)},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
            # Only print error if it's unexpected (not the alignment extension error)
            error_msg = str(e)
            if "alignment extension" not in error_msg.lower() and "not compiled" not in error_msg.lower():
                print(f"Error in forced alignment: {e}, using fallback")
            fallback_start = time.time()
            result = self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
            fallback_elapsed = (time.time() - fallback_start) * 1000
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:fallback_error","message":"Fallback alignment after error","data":{"segments_count":len(result),"error":str(e)},"timestamp":int(time.time()*1000),"elapsed_ms":int(fallback_elapsed)})+'\n')
            # #endregion
            return result
        
        # Calculate stride (time per frame)
        # Wav2Vec2 typically has downsampling factor of 320
        # So each frame corresponds to 320 samples at 16kHz = 20ms
        segment_extract_start = time.time()
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
        
        segment_extract_elapsed = (time.time() - segment_extract_start) * 1000
        total_elapsed = (time.time() - align_start) * 1000
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:after_segment_extract","message":"Segments extracted from alignment","data":{"segments_count":len(segments)},"timestamp":int(time.time()*1000),"elapsed_ms":int(segment_extract_elapsed)})+'\n')
        # #endregion
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"forced_alignment.py:extract_phoneme_segments:end","message":"Forced alignment extraction completed","data":{"total_elapsed_ms":int(total_elapsed),"segments_count":len(segments)},"timestamp":int(time.time()*1000),"elapsed_ms":int(total_elapsed)})+'\n')
        # #endregion
        
        return segments
    
    def _ctc_alignment(
        self,
        labels: List[str],
        emissions: torch.Tensor,
        dictionary: Dict[str, int],
        sample_rate: int,
        waveform: torch.Tensor,
        token_ids: List[int]
    ) -> List[PhonemeSegment]:
        """
        Greedy CTC-based forced alignment.
        Uses emissions to find the most likely alignment path for given phoneme sequence.
        
        Args:
            labels: List of phoneme labels
            emissions: Log-softmax emissions from model (batch, time, vocab_size)
            dictionary: Vocabulary mapping (token -> ID)
            sample_rate: Sample rate of audio
            waveform: Audio waveform tensor
            token_ids: List of token IDs corresponding to labels
            
        Returns:
            List of PhonemeSegment objects with accurate timing information
        """
        try:
            # Get best path using greedy CTC decode
            # emissions shape: (batch=1, time, vocab_size)
            emissions_np = emissions[0].cpu().numpy()  # (time, vocab_size)
            num_frames = emissions_np.shape[0]
            
            # Get most likely token at each time step
            best_path = np.argmax(emissions_np, axis=1)  # (time,)
            
            # Collapse repeated tokens and remove blanks (assuming blank_id=0 or '|')
            blank_id = self.blank_id
            
            # Find regions for each target phoneme
            segments = []
            target_idx = 0
            current_start = None
            
            for frame_idx in range(num_frames):
                pred_id = best_path[frame_idx]
                
                # Skip blank tokens
                if pred_id == blank_id:
                    if current_start is not None and target_idx < len(token_ids):
                        # End current segment
                        segments.append((target_idx, current_start, frame_idx - 1))
                        target_idx += 1
                        current_start = None
                    continue
                
                # Check if this is our target phoneme
                if target_idx < len(token_ids) and pred_id == token_ids[target_idx]:
                    if current_start is None:
                        current_start = frame_idx
                # If we encounter a different token, end current segment
                elif current_start is not None:
                    segments.append((target_idx, current_start, frame_idx - 1))
                    target_idx += 1
                    current_start = None
                    
                    # Check if new token matches next target
                    if target_idx < len(token_ids) and pred_id == token_ids[target_idx]:
                        current_start = frame_idx
            
            # Close last segment if needed
            if current_start is not None and target_idx < len(token_ids):
                segments.append((target_idx, current_start, num_frames - 1))
            
            # Calculate frame duration
            num_samples = waveform.shape[1]
            frame_duration = num_samples / num_frames / sample_rate
            
            # Convert to PhonemeSegment objects
            result_segments = []
            for phoneme_idx, start_frame, end_frame in segments:
                if phoneme_idx < len(labels):
                    label = labels[phoneme_idx]
                    start_time = start_frame * frame_duration
                    end_time = (end_frame + 1) * frame_duration
                    
                    # Calculate confidence score from emissions
                    frame_scores = []
                    for f in range(start_frame, min(end_frame + 1, num_frames)):
                        score = emissions_np[f, token_ids[phoneme_idx]]
                        frame_scores.append(score)
                    avg_score = np.mean(frame_scores) if frame_scores else -10.0
                    
                    result_segments.append(PhonemeSegment(
                        label=label,
                        start_time=start_time,
                        end_time=end_time,
                        score=float(avg_score),
                        frame_start=int(start_frame),
                        frame_end=int(end_frame)
                    ))
            
            # If we didn't find all phonemes, use fallback for missing ones
            if len(result_segments) < len(labels):
                print(f"Warning: CTC alignment found only {len(result_segments)}/{len(labels)} phonemes, using hybrid approach")
                return self._hybrid_alignment(labels, result_segments, emissions, dictionary, sample_rate, waveform)
            
            return result_segments
            
        except Exception as e:
            print(f"Warning: CTC alignment failed: {e}, using simple fallback")
            import traceback
            traceback.print_exc()
            # Fall back to simple time-based distribution
            return self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
    
    def _hybrid_alignment(
        self,
        labels: List[str],
        found_segments: List[PhonemeSegment],
        emissions: torch.Tensor,
        dictionary: Dict[str, int],
        sample_rate: int,
        waveform: torch.Tensor
    ) -> List[PhonemeSegment]:
        """
        Hybrid alignment: use CTC segments where found, interpolate missing ones.
        """
        if not found_segments:
            return self._fallback_alignment(labels, emissions, dictionary, sample_rate, waveform)
        
        # Get frame duration
        num_frames = emissions.shape[1]
        num_samples = waveform.shape[1]
        frame_duration = num_samples / num_frames / sample_rate
        total_duration = num_samples / sample_rate
        
        # Build complete segments list
        result = []
        found_idx = 0
        
        for i, label in enumerate(labels):
            # Check if this label was found by CTC
            if found_idx < len(found_segments) and found_segments[found_idx].label == label:
                result.append(found_segments[found_idx])
                found_idx += 1
            else:
                # Interpolate position
                if result:
                    # Place after last segment
                    last_seg = result[-1]
                    start_time = last_seg.end_time
                else:
                    start_time = 0.0
                
                # Estimate duration based on average
                avg_duration = total_duration / len(labels)
                end_time = min(start_time + avg_duration, total_duration)
                
                result.append(PhonemeSegment(
                    label=label,
                    start_time=start_time,
                    end_time=end_time,
                    score=0.3,  # Low confidence for interpolated
                    frame_start=int(start_time / frame_duration),
                    frame_end=int(end_time / frame_duration)
                ))
        
        return result
    
    def _fallback_alignment(
        self,
        labels: List[str],
        emissions: torch.Tensor,
        dictionary: Dict[str, int],
        sample_rate: int,
        waveform: torch.Tensor
    ) -> List[PhonemeSegment]:
        """
        Fallback alignment method when neither torchaudio.forced_align nor CTC aligner is available.
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


"""
Multi-level phoneme filtering module.
Combines Whitelist (German IPA + common errors) and Confidence Score filtering.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import config
import json


class PhonemeFilter:
    """Phoneme filter with whitelist and confidence filtering."""
    
    def __init__(
        self,
        whitelist: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        confidence_threshold_unclear: Optional[float] = None
    ):
        """
        Initialize phoneme filter.
        
        Args:
            whitelist: List of allowed phonemes. If None, uses config.PHONEME_WHITELIST
            confidence_threshold: Minimum confidence score (below this, phoneme is filtered out)
            confidence_threshold_unclear: Below this, mark as "unclear"
        """
        self.whitelist = set(whitelist or config.PHONEME_WHITELIST)
        self.confidence_threshold = confidence_threshold
        self.confidence_threshold_unclear = confidence_threshold_unclear or config.CONFIDENCE_THRESHOLD_UNCLEAR
    
    def filter_by_whitelist(self, phonemes: List[str]) -> List[str]:
        """
        Filter phonemes by whitelist.
        
        Args:
            phonemes: List of phoneme strings
            
        Returns:
            Filtered list of phonemes (only those in whitelist)
        """
        return [ph for ph in phonemes if ph in self.whitelist]
    
    def filter_by_confidence(
        self,
        logits: torch.Tensor,
        phonemes: List[str],
        vocab: Dict[str, int],
        whitelist: Optional[set] = None
    ) -> List[Dict]:
        """
        Filter phonemes by confidence score.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            phonemes: List of phoneme strings (from decoding) - used for whitelist check
            vocab: Vocabulary mapping (token -> ID)
            whitelist: Optional whitelist set to filter tokens
            
        Returns:
            List of dictionaries with filtered phonemes and metadata:
            [
                {
                    'phoneme': 'h',
                    'confidence': 0.85,
                    'is_unclear': False,
                    'frame_idx': 0
                },
                ...
            ]
        """
        # Compute probabilities from logits
        probs = torch.softmax(logits, dim=-1)  # (batch, time, vocab_size)
        
        # Get predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)  # (batch, time)
        
        # Create reverse vocab (ID -> token)
        id_to_token = {v: k for k, v in vocab.items()}
        
        # Use provided whitelist or default
        whitelist_set = whitelist if whitelist is not None else self.whitelist
        
        filtered_phonemes = []
        
        # Process each frame
        for frame_idx in range(predicted_ids.shape[1]):
            pred_id = predicted_ids[0, frame_idx].item()
            confidence = probs[0, frame_idx, pred_id].item()
            
            # Get phoneme token
            phoneme = id_to_token.get(pred_id, '')
            
            # Skip blank, PAD, and special tokens
            skip_tokens = ['[PAD]', '[UNK]', '<pad>', '<unk>', '|', 'h#', 'spn', '']
            if phoneme in skip_tokens or not phoneme:
                continue
            
            # Convert ARPAbet to IPA for consistent comparison with expected phonemes
            phoneme_ipa = config.convert_arpabet_to_ipa(phoneme)
            # Skip if conversion resulted in empty string (silence token)
            if not phoneme_ipa:
                continue
            
            # Skip if confidence too low
            if confidence < self.confidence_threshold:
                continue
            
            # Apply whitelist filter if provided (check both original and IPA)
            if whitelist_set and phoneme not in whitelist_set and phoneme_ipa not in whitelist_set:
                continue
            
            # Check if unclear
            is_unclear = confidence < self.confidence_threshold_unclear
            
            filtered_phonemes.append({
                'phoneme': phoneme_ipa,  # Store IPA version for consistent comparison
                'phoneme_arpabet': phoneme,  # Keep original for reference
                'confidence': confidence,
                'is_unclear': is_unclear,
                'frame_idx': frame_idx
            })
        
        return filtered_phonemes
    
    def _filter_isolated_phonemes(self, phonemes: List[Dict]) -> List[Dict]:
        """
        Filter out isolated phonemes that are likely noise.
        A phoneme is considered isolated if it's far from other phonemes.
        
        Args:
            phonemes: List of phoneme dictionaries with 'frame_idx' key
            
        Returns:
            Filtered list of phonemes
        """
        if len(phonemes) <= 2:
            return phonemes
        
        filtered = []
        frame_gap_threshold = 50  # Frames (approximately 1 second at 20ms per frame)
        
        for i, ph in enumerate(phonemes):
            frame_idx = ph.get('frame_idx', 0)
            
            # Check distance to previous and next phonemes
            prev_gap = float('inf')
            next_gap = float('inf')
            
            if i > 0:
                prev_frame = phonemes[i-1].get('frame_idx', 0)
                prev_gap = frame_idx - prev_frame
            
            if i < len(phonemes) - 1:
                next_frame = phonemes[i+1].get('frame_idx', 0)
                next_gap = next_frame - frame_idx
            
            # Keep phoneme if it's close to at least one neighbor
            # or if it has high confidence (likely real)
            min_gap = min(prev_gap, next_gap)
            confidence = ph.get('confidence', 0.0)
            
            if min_gap < frame_gap_threshold or confidence > 0.7:
                filtered.append(ph)
            # Otherwise, it's likely noise - skip it
        
        return filtered
    
    def filter_combined(
        self,
        logits: torch.Tensor,
        raw_phonemes: List[str],
        vocab: Dict[str, int]
    ) -> List[Dict]:
        """
        Combined filtering: first whitelist, then confidence.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            raw_phonemes: List of raw phoneme strings (from decoding)
            vocab: Vocabulary mapping (token -> ID)
            
        Returns:
            List of dictionaries with filtered phonemes and metadata
        """
        # Step 1: Whitelist filtering
        whitelisted_phonemes = self.filter_by_whitelist(raw_phonemes)
        
        # Step 2: Confidence filtering (with whitelist applied)
        confidence_filtered = self.filter_by_confidence(logits, whitelisted_phonemes, vocab, whitelist=self.whitelist)
        
        # Step 3: Post-processing - remove very short isolated phonemes (likely noise)
        # This helps reduce extra phonemes
        if len(confidence_filtered) > 0:
            confidence_filtered = self._filter_isolated_phonemes(confidence_filtered)
        
        return confidence_filtered


# Global instance
_phoneme_filter = None


def get_phoneme_filter(
    whitelist: Optional[List[str]] = None,
    confidence_threshold: Optional[float] = None
) -> PhonemeFilter:
    """Get or create global phoneme filter instance."""
    global _phoneme_filter
    if _phoneme_filter is None:
        confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        _phoneme_filter = PhonemeFilter(
            whitelist=whitelist,
            confidence_threshold=confidence_threshold
        )
    return _phoneme_filter


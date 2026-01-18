"""
Common utilities for phoneme processing.
"""

from typing import List


def collapse_consecutive_duplicates(phonemes: List[str]) -> List[str]:
    """
    Collapse consecutive duplicate phonemes (same logic as CTC collapse).
    This ensures that expected and recognized phonemes are processed consistently.
    
    Args:
        phonemes: List of phoneme strings
        
    Returns:
        List of phonemes with consecutive duplicates collapsed
    """
    if not phonemes:
        return phonemes
    
    collapsed = []
    prev_phoneme = None
    
    for phoneme in phonemes:
        # Skip empty phonemes
        if not phoneme or not phoneme.strip():
            continue
        
        # If different from previous, add it
        if phoneme != prev_phoneme:
            collapsed.append(phoneme)
            prev_phoneme = phoneme
        # If same as previous, skip it (CTC collapse)
        # prev_phoneme stays the same to allow same token later
    
    return collapsed

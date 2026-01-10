"""
Needleman-Wunsch alignment for matching expected and recognized phonemes.
"""

from typing import List, Tuple, Optional
import numpy as np

# Try to import biopython
try:
    from Bio import pairwise2
    from Bio.SubsMat import MatrixInfo as matlist
    HAS_BIO = True
except ImportError:
    HAS_BIO = False
    pairwise2 = None


def needleman_wunsch_align(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    gap_penalty: float = -1.0
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """
    Perform Needleman-Wunsch global alignment.
    
    Args:
        sequence1: First sequence (expected phonemes)
        sequence2: Second sequence (recognized phonemes)
        match_score: Score for matching characters
        mismatch_score: Score for mismatching characters
        gap_penalty: Penalty for gaps (should be negative)
        
    Returns:
        Tuple of (aligned_pairs, alignment_score):
        - aligned_pairs: List of tuples (seq1_char, seq2_char), where None indicates gap
        - alignment_score: Total alignment score
    """
    # Use biopython if available
    if HAS_BIO:
        return _align_biopython(sequence1, sequence2, match_score, mismatch_score, gap_penalty)
    else:
        return _align_manual(sequence1, sequence2, match_score, mismatch_score, gap_penalty)


def _align_biopython(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float,
    mismatch_score: float,
    gap_penalty: float
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """Alignment using biopython."""
    seq1_str = ''.join(sequence1)
    seq2_str = ''.join(sequence2)
    
    # Perform alignment
    alignments = pairwise2.align.globalms(
        seq1_str,
        seq2_str,
        match_score,
        mismatch_score,
        gap_penalty,
        gap_penalty
    )
    
    if not alignments:
        return ([], 0.0)
    
    # Get best alignment
    best_alignment = alignments[0]
    aligned_seq1 = best_alignment.seqA
    aligned_seq2 = best_alignment.seqB
    score = best_alignment.score
    
    # Convert to list of pairs
    pairs = []
    seq1_idx = 0
    seq2_idx = 0
    
    for i in range(len(aligned_seq1)):
        char1 = aligned_seq1[i]
        char2 = aligned_seq2[i]
        
        if char1 == '-':
            pairs.append((None, sequence2[seq2_idx] if seq2_idx < len(sequence2) else None))
            seq2_idx += 1
        elif char2 == '-':
            pairs.append((sequence1[seq1_idx] if seq1_idx < len(sequence1) else None, None))
            seq1_idx += 1
        else:
            pairs.append((
                sequence1[seq1_idx] if seq1_idx < len(sequence1) else None,
                sequence2[seq2_idx] if seq2_idx < len(sequence2) else None
            ))
            seq1_idx += 1
            seq2_idx += 1
    
    return (pairs, score)


def _align_manual(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float,
    mismatch_score: float,
    gap_penalty: float
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """Manual implementation of Needleman-Wunsch."""
    m = len(sequence1)
    n = len(sequence2)
    
    # Initialize DP table
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize first row and column (gap penalties)
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + (match_score if sequence1[i-1] == sequence2[j-1] else mismatch_score)
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
    
    # Backtrack to get alignment
    pairs = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sequence1[i-1] == sequence2[j-1]:
            # Match
            pairs.append((sequence1[i-1], sequence2[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + gap_penalty):
            # Delete (gap in sequence2)
            pairs.append((sequence1[i-1], None))
            i -= 1
        else:
            # Insert (gap in sequence1)
            pairs.append((None, sequence2[j-1]))
            j -= 1
    
    # Reverse to get correct order
    pairs.reverse()
    
    return (pairs, dp[m][n])



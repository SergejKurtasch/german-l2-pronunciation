"""
Needleman-Wunsch alignment for matching expected and recognized phonemes.
"""

from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from modules.phoneme_similarity import get_phoneme_similarity

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
    gap_penalty: float = -1.0,
    use_similarity_matrix: bool = True
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """
    Perform Needleman-Wunsch global alignment.
    
    Args:
        sequence1: First sequence (expected phonemes)
        sequence2: Second sequence (recognized phonemes)
        match_score: Score for matching characters
        mismatch_score: Score for mismatching characters
        gap_penalty: Penalty for gaps (should be negative)
        use_similarity_matrix: Use phoneme similarity matrix for scoring (default: True)
        
    Returns:
        Tuple of (aligned_pairs, alignment_score):
        - aligned_pairs: List of tuples (seq1_char, seq2_char), where None indicates gap
        - alignment_score: Total alignment score
    """
    # Use biopython if available
    if HAS_BIO:
        return _align_biopython(
            sequence1, sequence2, match_score, mismatch_score, gap_penalty, use_similarity_matrix
        )
    else:
        return _align_manual(
            sequence1, sequence2, match_score, mismatch_score, gap_penalty, use_similarity_matrix
        )


def _create_substitution_matrix(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float,
    mismatch_score: float
) -> Dict:
    """
    Create substitution matrix for biopython alignment.
    
    Args:
        sequence1: First sequence
        sequence2: Second sequence
        match_score: Score for exact matches
        mismatch_score: Default score for mismatches
        
    Returns:
        Dictionary mapping (char1, char2) -> similarity_score
    """
    # Get unique phonemes from both sequences
    unique_phonemes = set(sequence1 + sequence2)
    
    # Build substitution matrix
    matrix = {}
    for ph1 in unique_phonemes:
        for ph2 in unique_phonemes:
            if ph1 == ph2:
                matrix[(ph1, ph2)] = match_score
            else:
                # Use phoneme similarity
                similarity = get_phoneme_similarity(ph1, ph2)
                # Scale to reasonable range (similarity is -0.5 to 1.0)
                matrix[(ph1, ph2)] = similarity
    
    return matrix


def _align_biopython(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float,
    mismatch_score: float,
    gap_penalty: float,
    use_similarity_matrix: bool = True
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """Alignment using biopython."""
    seq1_str = ''.join(sequence1)
    seq2_str = ''.join(sequence2)
    
    # Perform alignment with or without similarity matrix
    if use_similarity_matrix:
        # Create custom substitution matrix
        substitution_matrix = _create_substitution_matrix(sequence1, sequence2, match_score, mismatch_score)
        alignments = pairwise2.align.globaldx(
            seq1_str,
            seq2_str,
            substitution_matrix,
            gap_penalty,
            gap_penalty
        )
    else:
        # Use simple match/mismatch scoring
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
    gap_penalty: float,
    use_similarity_matrix: bool = True
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """Manual implementation of Needleman-Wunsch with word boundary support."""
    m = len(sequence1)
    n = len(sequence2)
    
    # Define reduced gap penalty for word boundary marker
    boundary_gap_penalty = gap_penalty * 0.5  # Less penalty for inserting/deleting '||'
    
    # Initialize DP table
    dp = np.zeros((m + 1, n + 1))
    
    # Initialize first row and column (gap penalties)
    for i in range(1, m + 1):
        # Use reduced penalty if this phoneme is a word boundary
        penalty = boundary_gap_penalty if sequence1[i-1] == '||' else gap_penalty
        dp[i][0] = dp[i-1][0] + penalty
    for j in range(1, n + 1):
        # Use reduced penalty if this phoneme is a word boundary
        penalty = boundary_gap_penalty if sequence2[j-1] == '||' else gap_penalty
        dp[0][j] = dp[0][j-1] + penalty
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate match/mismatch score using similarity matrix or simple scoring
            if use_similarity_matrix:
                similarity = get_phoneme_similarity(sequence1[i-1], sequence2[j-1])
                match = dp[i-1][j-1] + similarity
            else:
                match = dp[i-1][j-1] + (match_score if sequence1[i-1] == sequence2[j-1] else mismatch_score)
            
            # Use reduced gap penalty for word boundaries
            delete_penalty = boundary_gap_penalty if sequence1[i-1] == '||' else gap_penalty
            insert_penalty = boundary_gap_penalty if sequence2[j-1] == '||' else gap_penalty
            
            delete = dp[i-1][j] + delete_penalty
            insert = dp[i][j-1] + insert_penalty
            dp[i][j] = max(match, delete, insert)
    
    # Backtrack to get alignment
    pairs = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            # Calculate what score led to current cell
            if use_similarity_matrix:
                similarity = get_phoneme_similarity(sequence1[i-1], sequence2[j-1])
                match_score_used = similarity
            else:
                match_score_used = match_score if sequence1[i-1] == sequence2[j-1] else mismatch_score
            
            # Calculate gap penalties used (may be reduced for boundaries)
            delete_penalty = boundary_gap_penalty if sequence1[i-1] == '||' else gap_penalty
            insert_penalty = boundary_gap_penalty if sequence2[j-1] == '||' else gap_penalty
            
            # Check which operation led to this cell
            if abs(dp[i][j] - (dp[i-1][j-1] + match_score_used)) < 1e-9:
                # Match/Mismatch (diagonal)
                pairs.append((sequence1[i-1], sequence2[j-1]))
                i -= 1
                j -= 1
            elif abs(dp[i][j] - (dp[i-1][j] + delete_penalty)) < 1e-9:
                # Delete (gap in sequence2)
                pairs.append((sequence1[i-1], None))
                i -= 1
            else:
                # Insert (gap in sequence1)
                pairs.append((None, sequence2[j-1]))
                j -= 1
        elif i > 0:
            # Only expected sequence left
            pairs.append((sequence1[i-1], None))
            i -= 1
        else:
            # Only recognized sequence left
            pairs.append((None, sequence2[j-1]))
            j -= 1
    
    # Reverse to get correct order
    pairs.reverse()
    
    return (pairs, dp[m][n])



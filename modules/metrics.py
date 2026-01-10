"""
Metrics module for calculating WER (Word Error Rate) and PER (Phoneme Error Rate).
"""

from typing import List, Dict, Tuple, Optional

# Try to import jiwer for WER calculation
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    jiwer = None


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, any]:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis texts.
    
    WER = (S + D + I) / N
    where:
    - S = substitutions (замены)
    - D = deletions (пропуски)
    - I = insertions (вставки)
    - N = total words in reference
    
    Args:
        reference: Expected/reference text
        hypothesis: Recognized/hypothesis text
        
    Returns:
        Dictionary with WER metrics:
        {
            'wer': float,  # Word Error Rate (0.0 = perfect, 1.0 = all wrong)
            'substitutions': int,
            'deletions': int,
            'insertions': int,
            'hits': int,  # Correct words
            'total_reference_words': int
        }
    """
    if not reference or not reference.strip():
        # If reference is empty, any hypothesis is 100% error
        hypothesis_words = len(hypothesis.split()) if hypothesis else 0
        return {
            'wer': 1.0 if hypothesis_words > 0 else 0.0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': hypothesis_words,
            'hits': 0,
            'total_reference_words': 0
        }
    
    if not hypothesis or not hypothesis.strip():
        # If hypothesis is empty, all reference words are deletions
        reference_words = len(reference.split())
        return {
            'wer': 1.0 if reference_words > 0 else 0.0,
            'substitutions': 0,
            'deletions': reference_words,
            'insertions': 0,
            'hits': 0,
            'total_reference_words': reference_words
        }
    
    # Use jiwer if available
    if HAS_JIWER:
        try:
            measures = jiwer.compute_measures(reference, hypothesis)
            return {
                'wer': measures['wer'],
                'substitutions': measures['substitutions'],
                'deletions': measures['deletions'],
                'insertions': measures['insertions'],
                'hits': measures['hits'],
                'total_reference_words': len(reference.split())
            }
        except Exception as e:
            print(f"Warning: jiwer calculation failed: {e}, using fallback")
            # Fall through to manual implementation
    
    # Fallback: manual implementation using edit distance
    return _calculate_wer_manual(reference, hypothesis)


def _calculate_wer_manual(reference: str, hypothesis: str) -> Dict[str, any]:
    """
    Manual implementation of WER using edit distance algorithm.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    m = len(ref_words)
    n = len(hyp_words)
    
    # Dynamic programming table for edit distance
    # dp[i][j] = minimum edits to transform ref_words[0:i] to hyp_words[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize: empty string to empty string = 0 edits
    # Empty string to j words = j insertions
    for j in range(n + 1):
        dp[0][j] = j
    
    # i words to empty string = i deletions
    for i in range(1, m + 1):
        dp[i][0] = i
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                # Match: no edit needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Choose minimum of: substitution, deletion, insertion
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1       # insertion
                )
    
    # Backtrack to count operations
    substitutions = deletions = insertions = hits = 0
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            # Match
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion
            deletions += 1
            i -= 1
        else:
            # Insertion
            insertions += 1
            j -= 1
    
    total_errors = substitutions + deletions + insertions
    wer = total_errors / m if m > 0 else 0.0
    
    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'hits': hits,
        'total_reference_words': m
    }


def calculate_per(aligned_pairs: List[Tuple[Optional[str], Optional[str]]]) -> Dict[str, any]:
    """
    Calculate Phoneme Error Rate (PER) from aligned phoneme pairs.
    
    PER = (S + D + I) / N
    where:
    - S = substitutions (замены фонем)
    - D = deletions (пропущенные фонемы)
    - I = insertions (лишние фонемы)
    - N = total expected phonemes
    
    Args:
        aligned_pairs: List of tuples (expected_phoneme, recognized_phoneme)
                      from Needleman-Wunsch alignment.
                      None indicates gap (missing or extra phoneme).
        
    Returns:
        Dictionary with PER metrics:
        {
            'per': float,  # Phoneme Error Rate (0.0 = perfect, 1.0 = all wrong)
            'substitutions': int,
            'deletions': int,
            'insertions': int,
            'total_expected': int,
            'total_recognized': int
        }
    """
    substitutions = 0
    deletions = 0
    insertions = 0
    total_expected = 0
    total_recognized = 0
    
    for expected, recognized in aligned_pairs:
        if expected is None:
            # Insertion: extra phoneme in recognized sequence
            insertions += 1
            total_recognized += 1
        elif recognized is None:
            # Deletion: missing phoneme in recognized sequence
            deletions += 1
            total_expected += 1
        else:
            # Both present
            total_expected += 1
            total_recognized += 1
            if expected != recognized:
                # Substitution: different phoneme
                substitutions += 1
    
    total_errors = substitutions + deletions + insertions
    per = total_errors / total_expected if total_expected > 0 else 0.0
    
    return {
        'per': per,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_expected': total_expected,
        'total_recognized': total_recognized
    }

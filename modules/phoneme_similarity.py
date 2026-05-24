"""
Phoneme similarity module for improved phoneme alignment.

This module provides phonetic feature-based similarity scoring for phonemes,
allowing the Needleman-Wunsch alignment algorithm to distinguish between
similar phonemes (e.g., p/b differ only in voicing) and dissimilar ones
(e.g., p/a are completely different).

Features considered:
- For consonants: place of articulation, manner of articulation, voicing
- For vowels: height, backness, rounding, length
- For diphthongs: component vowels
"""

from typing import Dict, Set, Tuple, Optional
import numpy as np


# === PHONEME FEATURE DEFINITIONS ===

# Consonant classification
CONSONANT_FEATURES = {
    # Plosives (stops)
    'p': {'type': 'consonant', 'manner': 'plosive', 'place': 'bilabial', 'voiced': False},
    'b': {'type': 'consonant', 'manner': 'plosive', 'place': 'bilabial', 'voiced': True},
    't': {'type': 'consonant', 'manner': 'plosive', 'place': 'alveolar', 'voiced': False},
    'd': {'type': 'consonant', 'manner': 'plosive', 'place': 'alveolar', 'voiced': True},
    'k': {'type': 'consonant', 'manner': 'plosive', 'place': 'velar', 'voiced': False},
    'kʰ': {'type': 'consonant', 'manner': 'plosive', 'place': 'velar', 'voiced': False, 'aspirated': True},
    'g': {'type': 'consonant', 'manner': 'plosive', 'place': 'velar', 'voiced': True},
    'ʔ': {'type': 'consonant', 'manner': 'plosive', 'place': 'glottal', 'voiced': False},
    
    # Fricatives
    'f': {'type': 'consonant', 'manner': 'fricative', 'place': 'labiodental', 'voiced': False},
    'v': {'type': 'consonant', 'manner': 'fricative', 'place': 'labiodental', 'voiced': True},
    's': {'type': 'consonant', 'manner': 'fricative', 'place': 'alveolar', 'voiced': False},
    'z': {'type': 'consonant', 'manner': 'fricative', 'place': 'alveolar', 'voiced': True},
    'ʃ': {'type': 'consonant', 'manner': 'fricative', 'place': 'postalveolar', 'voiced': False},
    'ʒ': {'type': 'consonant', 'manner': 'fricative', 'place': 'postalveolar', 'voiced': True},
    'ç': {'type': 'consonant', 'manner': 'fricative', 'place': 'palatal', 'voiced': False},
    'x': {'type': 'consonant', 'manner': 'fricative', 'place': 'velar', 'voiced': False},
    'h': {'type': 'consonant', 'manner': 'fricative', 'place': 'glottal', 'voiced': False},
    
    # Affricates
    'pf': {'type': 'consonant', 'manner': 'affricate', 'place': 'labiodental', 'voiced': False},
    'ts': {'type': 'consonant', 'manner': 'affricate', 'place': 'alveolar', 'voiced': False},
    'tʃ': {'type': 'consonant', 'manner': 'affricate', 'place': 'postalveolar', 'voiced': False},
    'dʒ': {'type': 'consonant', 'manner': 'affricate', 'place': 'postalveolar', 'voiced': True},
    
    # Nasals
    'm': {'type': 'consonant', 'manner': 'nasal', 'place': 'bilabial', 'voiced': True},
    'n': {'type': 'consonant', 'manner': 'nasal', 'place': 'alveolar', 'voiced': True},
    'ŋ': {'type': 'consonant', 'manner': 'nasal', 'place': 'velar', 'voiced': True},
    
    # Liquids
    'l': {'type': 'consonant', 'manner': 'lateral', 'place': 'alveolar', 'voiced': True},
    
    # R-sounds (various German and loanword R variants)
    'ʁ': {'type': 'consonant', 'manner': 'approximant', 'place': 'uvular', 'voiced': True},
    'ʀ': {'type': 'consonant', 'manner': 'trill', 'place': 'uvular', 'voiced': True},
    'r': {'type': 'consonant', 'manner': 'trill', 'place': 'alveolar', 'voiced': True},
    'ɾ': {'type': 'consonant', 'manner': 'tap', 'place': 'alveolar', 'voiced': True},
    'ɹ': {'type': 'consonant', 'manner': 'approximant', 'place': 'alveolar', 'voiced': True},
    
    # Glides/Approximants
    'j': {'type': 'consonant', 'manner': 'approximant', 'place': 'palatal', 'voiced': True},
}

# Vowel classification
VOWEL_FEATURES = {
    # Short vowels
    'a': {'type': 'vowel', 'height': 'open', 'backness': 'central', 'rounded': False, 'long': False},
    'ɛ': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': False, 'long': False},
    'e': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': False, 'long': False},
    'ə': {'type': 'vowel', 'height': 'mid', 'backness': 'central', 'rounded': False, 'long': False},
    'ɪ': {'type': 'vowel', 'height': 'near-close', 'backness': 'front', 'rounded': False, 'long': False},
    'i': {'type': 'vowel', 'height': 'close', 'backness': 'front', 'rounded': False, 'long': False},
    'ɔ': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounded': True, 'long': False},
    'o': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounded': True, 'long': False},
    'œ': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': True, 'long': False},
    'ø': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': True, 'long': False},
    'ʊ': {'type': 'vowel', 'height': 'near-close', 'backness': 'back', 'rounded': True, 'long': False},
    'u': {'type': 'vowel', 'height': 'close', 'backness': 'back', 'rounded': True, 'long': False},
    'ʏ': {'type': 'vowel', 'height': 'near-close', 'backness': 'front', 'rounded': True, 'long': False},
    'y': {'type': 'vowel', 'height': 'close', 'backness': 'front', 'rounded': True, 'long': False},
    'ɐ': {'type': 'vowel', 'height': 'mid', 'backness': 'central', 'rounded': False, 'long': False},
    
    # Long vowels
    'aː': {'type': 'vowel', 'height': 'open', 'backness': 'central', 'rounded': False, 'long': True},
    'eː': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': False, 'long': True},
    'iː': {'type': 'vowel', 'height': 'close', 'backness': 'front', 'rounded': False, 'long': True},
    'oː': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounded': True, 'long': True},
    'øː': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounded': True, 'long': True},
    'uː': {'type': 'vowel', 'height': 'close', 'backness': 'back', 'rounded': True, 'long': True},
    'yː': {'type': 'vowel', 'height': 'close', 'backness': 'front', 'rounded': True, 'long': True},
}

# Diphthong classification
DIPHTHONG_FEATURES = {
    'aɪ̯': {'type': 'diphthong', 'start': 'a', 'end': 'ɪ'},
    'aʊ̯': {'type': 'diphthong', 'start': 'a', 'end': 'ʊ'},
    'ɔʏ̯': {'type': 'diphthong', 'start': 'ɔ', 'end': 'ʏ'},
}

# Combine all features
ALL_PHONEME_FEATURES = {**CONSONANT_FEATURES, **VOWEL_FEATURES, **DIPHTHONG_FEATURES}


# === FEATURE WEIGHTS ===

# Default weights for feature comparison
DEFAULT_WEIGHTS = {
    'voicing': 0.25,      # Voicing difference (p vs b)
    'place': 0.35,        # Place of articulation (p vs t vs k)
    'manner': 0.30,       # Manner of articulation (p vs f vs m)
    'length': 0.10,       # Vowel length (a vs aː)
    'height': 0.30,       # Vowel height (a vs e vs i)
    'backness': 0.35,     # Vowel backness (i vs u)
    'rounding': 0.20,     # Vowel rounding (i vs y)
}


# === SIMILARITY CALCULATION ===

def calculate_consonant_similarity(
    features1: Dict,
    features2: Dict,
    weights: Optional[Dict] = None
) -> float:
    """
    Calculate similarity between two consonants based on phonetic features.
    
    Args:
        features1: Features of first consonant
        features2: Features of second consonant
        weights: Feature weights (optional)
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    similarity = 0.0
    
    # Same manner of articulation
    if features1['manner'] == features2['manner']:
        similarity += weights['manner']
    
    # Same place of articulation
    if features1['place'] == features2['place']:
        similarity += weights['place']
    
    # Same voicing
    if features1['voiced'] == features2['voiced']:
        similarity += weights['voicing']
    
    # Special handling for aspiration (kʰ vs k)
    if features1.get('aspirated', False) == features2.get('aspirated', False):
        similarity += 0.05  # Small bonus for matching aspiration
    
    return similarity


def calculate_vowel_similarity(
    features1: Dict,
    features2: Dict,
    weights: Optional[Dict] = None
) -> float:
    """
    Calculate similarity between two vowels based on phonetic features.
    
    Args:
        features1: Features of first vowel
        features2: Features of second vowel
        weights: Feature weights (optional)
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    similarity = 0.0
    
    # Same height
    if features1['height'] == features2['height']:
        similarity += weights['height']
    elif abs(_vowel_height_distance(features1['height'], features2['height'])) == 1:
        # Adjacent heights (e.g., close vs near-close)
        similarity += weights['height'] * 0.5
    
    # Same backness
    if features1['backness'] == features2['backness']:
        similarity += weights['backness']
    elif _vowel_backness_adjacent(features1['backness'], features2['backness']):
        similarity += weights['backness'] * 0.5
    
    # Same rounding
    if features1['rounded'] == features2['rounded']:
        similarity += weights['rounding']
    
    # Same length (very important for German)
    if features1['long'] == features2['long']:
        similarity += weights['length']
    
    return similarity


def calculate_diphthong_similarity(
    features1: Dict,
    features2: Dict
) -> float:
    """
    Calculate similarity between two diphthongs.
    
    Args:
        features1: Features of first diphthong
        features2: Features of second diphthong
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Compare starting and ending vowels
    start_sim = get_phoneme_similarity(features1['start'], features2['start'])
    end_sim = get_phoneme_similarity(features1['end'], features2['end'])
    
    # Average similarity
    return (start_sim + end_sim) / 2.0


def _vowel_height_distance(height1: str, height2: str) -> int:
    """Calculate distance between vowel heights."""
    height_order = ['close', 'near-close', 'mid', 'open']
    try:
        idx1 = height_order.index(height1)
        idx2 = height_order.index(height2)
        return abs(idx1 - idx2)
    except ValueError:
        return 999  # Unknown height


def _vowel_backness_adjacent(backness1: str, backness2: str) -> bool:
    """Check if two vowel backness values are adjacent."""
    backness_order = ['front', 'central', 'back']
    try:
        idx1 = backness_order.index(backness1)
        idx2 = backness_order.index(backness2)
        return abs(idx1 - idx2) == 1
    except ValueError:
        return False


def get_phoneme_similarity(
    phoneme1: str,
    phoneme2: str,
    weights: Optional[Dict] = None
) -> float:
    """
    Calculate similarity between two phonemes.
    
    Similarity scale:
        1.0: Identical phonemes
        0.8-0.9: Very similar (same phoneme, different length)
        0.6-0.7: Similar (same class, one feature differs)
        0.3-0.5: Somewhat similar (same class, multiple features differ)
        0.0-0.2: Different class
        -0.5: Very different (vowel vs consonant)
        -2.0: Word boundary mismatch (one has '||', other doesn't)
    
    Special handling for word boundary marker '||':
        - Both '||': perfect match (1.0)
        - One '||', one phoneme: large penalty (-2.0) to discourage boundary misalignment
        - One '||', one None: treated as gap (handled by alignment algorithm)
    
    Args:
        phoneme1: First phoneme (IPA string, or '||' for word boundary)
        phoneme2: Second phoneme (IPA string, or '||' for word boundary)
        weights: Feature weights (optional)
        
    Returns:
        Similarity score (-2.0 to 1.0)
    """
    # Identical phonemes
    if phoneme1 == phoneme2:
        return 1.0
    
    # Word boundary marker handling
    if phoneme1 == '||' or phoneme2 == '||':
        # Both are boundaries (already handled above with ==)
        # One is boundary, other is not: strong penalty
        return -2.0
    
    # Special rules for frequent confusions (BEFORE normalization)
    # These pairs are phonetically close and should have increased similarity
    # Note: Check BEFORE normalization to handle ARPABET 'r' correctly
    HIGH_SIMILARITY_PAIRS = {
        ('ɐ', 'ɾ'): 0.7,  # Central vowel and alveolar tap - phonetically close
        ('eː', 'ɛ'): 0.8,  # Same vowel, different length
        ('ɐ', 'ɜ'): 0.75,  # Both central vowels
        ('ʁ', 'ɾ'): 0.65,  # Both R-sounds, different places of articulation
        ('ʁ', 'ʀ'): 0.90,  # Both uvular R-sounds, different manner
        ('ɾ', 'r'): 0.85,  # Both alveolar R-sounds, different manner
        ('r', 'ɹ'): 0.80,  # Alveolar trill and approximant
    }
    
    # Check in both directions
    pair_key = (phoneme1, phoneme2)
    reverse_pair_key = (phoneme2, phoneme1)
    
    if pair_key in HIGH_SIMILARITY_PAIRS:
        return HIGH_SIMILARITY_PAIRS[pair_key]
    if reverse_pair_key in HIGH_SIMILARITY_PAIRS:
        return HIGH_SIMILARITY_PAIRS[reverse_pair_key]
    
    # Special rules for affricates vs fricatives
    # s → ts is often an artifact, but they are phonetically different
    if (phoneme1 == 's' and phoneme2 == 'ts') or (phoneme1 == 'ts' and phoneme2 == 's'):
        return 0.3  # Low similarity, as these are different types of sounds
    
    # Convert ARPABET to IPA if needed
    phoneme1 = _normalize_phoneme(phoneme1)
    phoneme2 = _normalize_phoneme(phoneme2)
    
    # Check again after normalization
    if phoneme1 == phoneme2:
        return 1.0
    
    # Get features
    features1 = ALL_PHONEME_FEATURES.get(phoneme1)
    features2 = ALL_PHONEME_FEATURES.get(phoneme2)
    
    # Unknown phonemes
    if features1 is None or features2 is None:
        return 0.0 if phoneme1 != phoneme2 else 1.0
    
    # Different types (consonant vs vowel)
    if features1['type'] != features2['type']:
        return -0.5
    
    # Calculate similarity based on type
    if features1['type'] == 'consonant':
        return calculate_consonant_similarity(features1, features2, weights)
    elif features1['type'] == 'vowel':
        return calculate_vowel_similarity(features1, features2, weights)
    elif features1['type'] == 'diphthong':
        return calculate_diphthong_similarity(features1, features2)
    
    return 0.0


def _normalize_phoneme(phoneme: str) -> str:
    """
    Normalize phoneme (convert ARPABET to IPA if needed).
    
    Args:
        phoneme: Phoneme string
        
    Returns:
        Normalized IPA phoneme
    """
    # Import here to avoid circular dependency
    try:
        import config
        return config.convert_arpabet_to_ipa(phoneme)
    except (ImportError, AttributeError):
        # If config not available, return as-is
        return phoneme


# === PRECOMPUTED SIMILARITY MATRIX ===

_similarity_matrix_cache: Optional[Dict[Tuple[str, str], float]] = None


def get_similarity_matrix(phoneme_list: Optional[list] = None) -> Dict[Tuple[str, str], float]:
    """
    Get precomputed similarity matrix for given phonemes.
    
    Args:
        phoneme_list: List of phonemes to compute matrix for (optional)
        
    Returns:
        Dictionary mapping (phoneme1, phoneme2) -> similarity_score
    """
    global _similarity_matrix_cache
    
    if phoneme_list is None:
        # Use all known phonemes
        phoneme_list = list(ALL_PHONEME_FEATURES.keys())
    
    # Check cache
    if _similarity_matrix_cache is not None:
        return _similarity_matrix_cache
    
    # Compute matrix
    matrix = {}
    for ph1 in phoneme_list:
        for ph2 in phoneme_list:
            similarity = get_phoneme_similarity(ph1, ph2)
            matrix[(ph1, ph2)] = similarity
    
    _similarity_matrix_cache = matrix
    return matrix


def clear_similarity_cache():
    """Clear the similarity matrix cache."""
    global _similarity_matrix_cache
    _similarity_matrix_cache = None


# === UTILITY FUNCTIONS ===

def get_phoneme_type(phoneme: str) -> str:
    """
    Get phoneme type (consonant, vowel, or diphthong).
    
    Args:
        phoneme: Phoneme string
        
    Returns:
        Type string ('consonant', 'vowel', 'diphthong', or 'unknown')
    """
    phoneme = _normalize_phoneme(phoneme)
    features = ALL_PHONEME_FEATURES.get(phoneme)
    return features['type'] if features else 'unknown'


def get_similar_phonemes(
    phoneme: str,
    threshold: float = 0.6,
    phoneme_list: Optional[list] = None
) -> list:
    """
    Get list of phonemes similar to the given phoneme.
    
    Args:
        phoneme: Target phoneme
        threshold: Minimum similarity threshold
        phoneme_list: List of phonemes to compare with (optional)
        
    Returns:
        List of tuples (phoneme, similarity_score) sorted by similarity
    """
    if phoneme_list is None:
        phoneme_list = list(ALL_PHONEME_FEATURES.keys())
    
    similar = []
    for other in phoneme_list:
        if other == phoneme:
            continue
        similarity = get_phoneme_similarity(phoneme, other)
        if similarity >= threshold:
            similar.append((other, similarity))
    
    # Sort by similarity (descending)
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar

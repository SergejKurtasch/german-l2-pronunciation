"""
Phoneme validation utilities for parallel processing.
"""

from typing import Dict, Any, Optional


def validate_single_phoneme(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function for parallel phoneme validation.
    Uses validate_phoneme() with full audio waveform and position_ms.

    Args:
        task_data: Dictionary containing validation task parameters:
            - audio: Full audio waveform
            - phoneme: Recognized phoneme (what was detected)
            - position_ms: Position in milliseconds where the phoneme is located
            - expected_phoneme: Expected phoneme
            - index: Index in aligned_pairs
            - segment_index: Index in recognized_segments
            - validator: Phoneme validator instance

    Returns:
        Dictionary with validation result and metadata:
            - index: Original index in aligned_pairs
            - segment_index: Index in recognized_segments
            - validation_result: Result from validate_phoneme
            - expected_phoneme: Expected phoneme
            - recognized_phoneme: Recognized phoneme
            - phoneme_pair: Phoneme pair string
            - error: Error message if any
    """
    try:
        validator = task_data['validator']
        result = validator.validate_phoneme(
            audio=task_data['audio'],
            phoneme=task_data['phoneme'],
            position_ms=task_data['position_ms'],
            expected_phoneme=task_data['expected_phoneme']
        )
        return {
            'index': task_data['index'],
            'segment_index': task_data['segment_index'],
            'validation_result': result,
            'expected_phoneme': task_data['expected_phoneme'],
            'recognized_phoneme': task_data['phoneme'],
            'phoneme_pair': task_data.get('phoneme_pair', ''),
            'error': None
        }
    except Exception as e:
        return {
            'index': task_data['index'],
            'segment_index': task_data.get('segment_index', -1),
            'validation_result': {
                'is_correct': None,
                'confidence': 0.0,
                'error': str(e)
            },
            'expected_phoneme': task_data.get('expected_phoneme', ''),
            'recognized_phoneme': task_data.get('phoneme', ''),
            'phoneme_pair': task_data.get('phoneme_pair', ''),
            'error': str(e)
        }

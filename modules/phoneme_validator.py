"""
Optional phoneme validator module for additional validation through trained models.
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path
import sys

# Try to import validator from main project (optional)
try:
    # Add parent project to path if available
    parent_project = Path(__file__).parent.parent.parent / "SpeechRec-German"
    if parent_project.exists():
        sys.path.insert(0, str(parent_project))
        from gradio_modules.phoneme_validator import PhonemeValidator, get_validator
        HAS_VALIDATOR = True
    else:
        HAS_VALIDATOR = False
        PhonemeValidator = None
        get_validator = None
except ImportError:
    HAS_VALIDATOR = False
    PhonemeValidator = None
    get_validator = None


class OptionalPhonemeValidator:
    """Wrapper for optional phoneme validation."""
    
    def __init__(self):
        """Initialize optional validator."""
        self.validator = None
        if HAS_VALIDATOR:
            try:
                self.validator = get_validator()
                print("Optional phoneme validator loaded from main project")
            except Exception as e:
                print(f"Warning: Failed to load optional validator: {e}")
                self.validator = None
    
    def has_trained_model(
        self,
        expected_phoneme: str,
        recognized_phoneme: str
    ) -> bool:
        """
        Check if trained model exists for phoneme pair.
        
        Args:
            expected_phoneme: Expected phoneme
            recognized_phoneme: Recognized phoneme
            
        Returns:
            True if model exists, False otherwise
        """
        if self.validator is None:
            return False
        
        try:
            return self.validator.has_trained_model(expected_phoneme, recognized_phoneme)
        except Exception as e:
            print(f"Error checking for trained model: {e}")
            return False
    
    def validate_phoneme_segment(
        self,
        audio_segment: np.ndarray,
        phoneme_pair: str,
        expected_phoneme: str,
        suspected_phoneme: str,
        sr: int
    ) -> Dict:
        """
        Validate phoneme segment using trained model.
        
        Args:
            audio_segment: Audio segment as numpy array
            phoneme_pair: Phoneme pair name (e.g., 'b-p')
            expected_phoneme: Expected phoneme
            suspected_phoneme: Suspected phoneme
            sr: Sample rate
            
        Returns:
            Dictionary with validation results:
            {
                'is_correct': bool,
                'confidence': float,
                'predicted_phoneme': str,
                ...
            }
        """
        if self.validator is None:
            return {
                'is_correct': None,
                'confidence': 0.0,
                'error': 'Validator not available'
            }
        
        try:
            result = self.validator.validate_phoneme_segment(
                audio_segment,
                phoneme_pair=phoneme_pair,
                expected_phoneme=expected_phoneme,
                suspected_phoneme=suspected_phoneme,
                sr=sr
            )
            return result
        except Exception as e:
            print(f"Error in phoneme validation: {e}")
            return {
                'is_correct': None,
                'confidence': 0.0,
                'error': str(e)
            }


# Global instance
_optional_validator = None


def get_optional_validator() -> OptionalPhonemeValidator:
    """Get or create global optional validator instance."""
    global _optional_validator
    if _optional_validator is None:
        _optional_validator = OptionalPhonemeValidator()
    return _optional_validator



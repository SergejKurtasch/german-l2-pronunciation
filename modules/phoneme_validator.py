"""
Optional phoneme validator module for additional validation through trained models.
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path

# Try to import validator from installed german-phoneme-validator package (optional)
try:
    from german_phoneme_validator import PhonemeValidator, get_validator
    HAS_VALIDATOR = True
    print("Found german-phoneme-validator package (installed via pip/conda)")
except ImportError as e:
    HAS_VALIDATOR = False
    PhonemeValidator = None
    get_validator = None
    print(f"german-phoneme-validator package not found. Install it with: pip install -e /path/to/german-phoneme-validator")
    print(f"Import error: {e}")


class OptionalPhonemeValidator:
    """Wrapper for optional phoneme validation."""
    
    def __init__(self):
        """Initialize optional validator."""
        self.validator = None
        if HAS_VALIDATOR:
            try:
                # Initialize validator (artifacts_dir will be auto-detected from installed package)
                self.validator = get_validator()
                available_pairs = self.validator.get_available_pairs()
                print(f"Optional phoneme validator loaded from german-phoneme-validator package")
                print(f"Available phoneme pairs: {len(available_pairs)} pairs")
                if available_pairs:
                    print(f"Sample pairs: {', '.join(available_pairs[:5])}")
                
                # Apply monkey-patch to fix vot_category issue
                self._apply_feature_extraction_fix()
            except Exception as e:
                print(f"Warning: Failed to load optional validator: {e}")
                import traceback
                traceback.print_exc()
                self.validator = None
    
    def _apply_feature_extraction_fix(self):
        """
        Apply monkey-patch to fix extract_vot function.
        Removes vot_category (categorical string) from features to prevent float conversion error.
        """
        try:
            # Import the feature extraction module from installed package
            from german_phoneme_validator.core import feature_extraction
            
            # Save original function
            original_extract_vot = feature_extraction.extract_vot
            
            # Create patched version
            def patched_extract_vot(audio, sr=16000):
                """Patched version that excludes vot_category."""
                result = original_extract_vot(audio, sr)
                # Remove categorical feature to prevent string->float conversion error
                if 'vot_category' in result:
                    del result['vot_category']
                return result
            
            # Apply patch
            feature_extraction.extract_vot = patched_extract_vot
            print("Applied feature extraction fix: removed vot_category from extract_vot")
        except Exception as e:
            print(f"Warning: Could not apply feature extraction fix: {e}")
    
    def get_phoneme_pair(
        self,
        expected_phoneme: str,
        recognized_phoneme: str
    ) -> Optional[str]:
        """
        Get phoneme pair name for two phonemes.
        
        Args:
            expected_phoneme: Expected phoneme
            recognized_phoneme: Recognized phoneme
            
        Returns:
            Phoneme pair name (e.g., 'b-p') or None if not found
        """
        if self.validator is None:
            return None
        
        try:
            return self.validator.get_phoneme_pair(expected_phoneme, recognized_phoneme)
        except Exception as e:
            print(f"Error getting phoneme pair: {e}")
            return None
    
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
    
    def validate_phoneme(
        self,
        audio: np.ndarray,
        phoneme: str,
        position_ms: float,
        expected_phoneme: str
    ) -> Dict:
        """
        Validate phoneme using trained model with full audio waveform and position.
        This method matches the notebook implementation.
        
        Args:
            audio: Full audio waveform as numpy array
            phoneme: Recognized phoneme (what was detected)
            position_ms: Position in milliseconds where the phoneme is located
            expected_phoneme: Expected correct phoneme
            
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
            result = self.validator.validate_phoneme(
                audio=audio,
                phoneme=phoneme,
                position_ms=position_ms,
                expected_phoneme=expected_phoneme
            )
            
            return result
        except Exception as e:
            print(f"Error in phoneme validation: {e}")
            return {
                'is_correct': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
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



"""
Diagnostic Engine for generating feedback based on pronunciation errors.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import config


class DiagnosticEngine:
    """Engine for generating diagnostic feedback."""
    
    def __init__(self, matrix_path: Optional[Path] = None):
        """
        Initialize diagnostic engine.
        
        Args:
            matrix_path: Path to diagnostic_matrix.json. If None, uses config default.
        """
        if matrix_path is None:
            matrix_path = config.DIAGNOSTIC_MATRIX_PATH
        
        self.matrix_path = Path(matrix_path)
        self.error_matrix = {}
        self.default_feedback = {}
        
        self._load_matrix()
    
    def _load_matrix(self):
        """Load diagnostic matrix from JSON file."""
        try:
            with open(self.matrix_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load error mappings
            self.error_matrix = {}
            for error in data.get('errors', []):
                target = error.get('target', '')
                user = error.get('user', '')
                key = f"{target}->{user}"
                self.error_matrix[key] = error
            
            # Load default feedback
            self.default_feedback = {
                'en': data.get('default_feedback_en', 'Try to match the target sound more closely.'),
                'ru': data.get('default_feedback_ru', 'Постарайтесь точнее соответствовать целевому звуку.'),
                'de': data.get('default_feedback_de', 'Versuchen Sie, dem Zielklang näher zu kommen.')
            }
            
            print(f"Loaded diagnostic matrix with {len(self.error_matrix)} error mappings")
            
        except Exception as e:
            print(f"Warning: Failed to load diagnostic matrix: {e}")
            self.error_matrix = {}
            self.default_feedback = {
                'en': 'Try to match the target sound more closely.',
                'ru': 'Постарайтесь точнее соответствовать целевому звуку.',
                'de': 'Versuchen Sie, dem Zielklang näher zu kommen.'
            }
    
    def get_feedback(
        self,
        target_phoneme: str,
        user_phoneme: str,
        language: str = 'en'
    ) -> Optional[str]:
        """
        Get feedback for a specific error.
        
        Args:
            target_phoneme: Expected phoneme
            user_phoneme: User's phoneme
            language: Language for feedback ('en', 'ru', 'de')
            
        Returns:
            Feedback string, or None if no error
        """
        # Check if it's a match (no error)
        if target_phoneme == user_phoneme:
            return None
        
        # Look up in error matrix
        key = f"{target_phoneme}->{user_phoneme}"
        error_entry = self.error_matrix.get(key)
        
        if error_entry:
            feedback_key = f"feedback_{language}"
            return error_entry.get(feedback_key, self.default_feedback.get(language, ''))
        
        # Return default feedback
        return self.default_feedback.get(language, '')
    
    def analyze_pronunciation(
        self,
        aligned_pairs: List[Tuple[Optional[str], Optional[str]]]
    ) -> List[Dict]:
        """
        Analyze pronunciation errors from aligned pairs.
        
        Args:
            aligned_pairs: List of tuples (expected_phoneme, recognized_phoneme)
            
        Returns:
            List of dictionaries with error analysis:
            [
                {
                    'expected': 'ʏ',
                    'recognized': 'u',
                    'is_correct': False,
                    'is_missing': False,
                    'is_extra': False,
                    'feedback_en': '...',
                    ...
                },
                ...
            ]
        """
        results = []
        
        for expected, recognized in aligned_pairs:
            # Check for gaps (missing or extra phonemes)
            if expected is None:
                # Extra phoneme (not in expected)
                results.append({
                    'expected': None,
                    'recognized': recognized,
                    'is_correct': False,
                    'is_missing': False,
                    'is_extra': True,
                    'feedback_en': f"Extra phoneme '{recognized}' detected.",
                    'feedback_ru': f"Обнаружена лишняя фонема '{recognized}'.",
                    'feedback_de': f"Zusätzliches Phonem '{recognized}' erkannt."
                })
            elif recognized is None:
                # Missing phoneme
                results.append({
                    'expected': expected,
                    'recognized': None,
                    'is_correct': False,
                    'is_missing': True,
                    'is_extra': False,
                    'feedback_en': f"Missing phoneme '{expected}'. Try to pronounce it.",
                    'feedback_ru': f"Пропущена фонема '{expected}'. Попробуйте произнести её.",
                    'feedback_de': f"Fehlendes Phonem '{expected}'. Versuchen Sie, es auszusprechen."
                })
            else:
                # Both present - check if match
                is_correct = (expected == recognized)
                feedback_en = self.get_feedback(expected, recognized, 'en')
                feedback_ru = self.get_feedback(expected, recognized, 'ru')
                feedback_de = self.get_feedback(expected, recognized, 'de')
                
                results.append({
                    'expected': expected,
                    'recognized': recognized,
                    'is_correct': is_correct,
                    'is_missing': False,
                    'is_extra': False,
                    'feedback_en': feedback_en or '',
                    'feedback_ru': feedback_ru or '',
                    'feedback_de': feedback_de or ''
                })
        
        return results


# Global instance
_diagnostic_engine = None


def get_diagnostic_engine(matrix_path: Optional[Path] = None) -> DiagnosticEngine:
    """Get or create global diagnostic engine instance."""
    global _diagnostic_engine
    if _diagnostic_engine is None:
        _diagnostic_engine = DiagnosticEngine(matrix_path=matrix_path)
    return _diagnostic_engine



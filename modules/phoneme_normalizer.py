"""
Phoneme normalization module using phoneme_normalization_table.json.
Applies Unicode normalization and removes diacritics/symbols not in model vocabulary.
"""

import json
from pathlib import Path
from typing import List, Set, Dict, Optional
import unicodedata


class PhonemeNormalizer:
    """Normalizes phonemes according to phoneme_normalization_table.json strategy."""
    
    def __init__(self, table_path: Optional[Path] = None):
        """
        Initialize phoneme normalizer.
        
        Args:
            table_path: Path to phoneme_normalization_table.json. If None, uses default.
        """
        if table_path is None:
            # Default path: project root / phoneme_normalization_table.json
            project_root = Path(__file__).parent.parent
            table_path = project_root / "phoneme_normalization_table.json"
        
        self.table_path = Path(table_path)
        self.normalization_table: Dict = {}
        self.phoneme_mapping: Dict[str, str] = {}
        self.diacritics_to_remove: Set[str] = set()
        self.suprasegmentals_to_remove: Set[str] = set()
        self.chars_to_remove: Set[str] = set()
        
        self._load_table()
    
    def _load_table(self):
        """Load normalization table from JSON file."""
        try:
            if not self.table_path.exists():
                print(f"Warning: Normalization table not found at {self.table_path}")
                print("Phoneme normalization will be skipped.")
                return
            
            with open(self.table_path, 'r', encoding='utf-8') as f:
                self.normalization_table = json.load(f)
            
            # Extract phoneme mapping (only g -> ɡ)
            self.phoneme_mapping = self.normalization_table.get('phoneme_mapping', {})
            
            # Extract diacritics to remove (decision == 'remove')
            diacritics = self.normalization_table.get('diacritics', {})
            for char, info in diacritics.items():
                if info.get('decision') == 'remove':
                    self.diacritics_to_remove.add(char)
            
            # Extract suprasegmentals to remove (decision == 'remove')
            suprasegmentals = self.normalization_table.get('suprasegmentals', {})
            for char, info in suprasegmentals.items():
                if info.get('decision') == 'remove':
                    self.suprasegmentals_to_remove.add(char)
            
            # Extract characters to remove from dictionaries
            chars_to_remove_list = self.normalization_table.get('chars_to_remove_from_dicts', [])
            self.chars_to_remove = {item.get('char', '') for item in chars_to_remove_list if item.get('char')}
            
            # Extract invalid patterns for validation
            invalid_patterns = self.normalization_table.get('invalid_patterns', [])
            
            print(f"Loaded phoneme normalization table from {self.table_path}")
            print(f"  - Phoneme mappings: {len(self.phoneme_mapping)}")
            print(f"  - Diacritics to remove: {len(self.diacritics_to_remove)}")
            print(f"  - Suprasegmentals to remove: {len(self.suprasegmentals_to_remove)}")
            print(f"  - Invalid patterns: {len(invalid_patterns)}")
            print(f"  - Characters to remove: {len(self.chars_to_remove)}")
            
        except Exception as e:
            print(f"Error loading normalization table: {e}")
            print("Phoneme normalization will be skipped.")
    
    def _is_valid_transcription(self, transcription: str) -> bool:
        """
        Validate transcription for invalid patterns.
        Returns False if transcription contains OCR errors or corrupted data.
        
        Args:
            transcription: Raw transcription string to validate
            
        Returns:
            True if valid, False if contains invalid patterns
        """
        if not self.normalization_table:
            return True
        
        invalid_patterns = self.normalization_table.get('invalid_patterns', [])
        
        import re
        for pattern in invalid_patterns:
            if re.search(pattern, transcription):
                return False
        
        # Check for suspicious capital letter sequences (except model vocabulary)
        allowed_caps = {'N', 'S', 'X', 'Z'}
        caps_sequence = re.findall(r'[A-Z]+', transcription)
        for seq in caps_sequence:
            if len(seq) >= 3 or (len(seq) >= 1 and set(seq) - allowed_caps):
                return False
        
        return True
    
    def normalize_phoneme_string(self, phoneme_string: str, source: str = 'dictionary') -> str:
        """
        Normalize a phoneme string according to the normalization table.
        
        This function applies:
        1. Validation (check for invalid patterns)
        2. Unicode normalization (NFC)
        3. Phoneme mapping (g -> ɡ, ˑ -> ː, etc.)
        4. Remove combining characters from affricates (t͡s -> ts, t͜s -> ts)
        5. Remove diacritics not in model
        6. Remove suprasegmentals not in model
        7. Remove characters not in model vocabulary
        
        Args:
            phoneme_string: Phoneme string to normalize
            source: Source of phonemes ('dictionary' or 'model'). 
                    Only 'dictionary' sources are normalized (model phonemes are kept as-is).
        
        Returns:
            Normalized phoneme string (empty string if invalid)
        """
        # Strategy: Only normalize dictionary sources, not model sources
        if source != 'dictionary':
            # For model sources, only apply Unicode NFC normalization (no character removal)
            return unicodedata.normalize('NFC', phoneme_string)
        
        if not self.normalization_table:
            # If table not loaded, return as-is
            return phoneme_string
        
        # Step 0: Validate before normalization (only for dictionary sources)
        if source == 'dictionary' and not self._is_valid_transcription(phoneme_string):
            return ""  # Return empty string for invalid transcriptions
        
        # Step 1: Unicode normalization (NFC)
        normalized = unicodedata.normalize('NFC', phoneme_string)
        
        # Step 2: Apply phoneme mapping (g -> ɡ)
        for from_char, to_char in self.phoneme_mapping.items():
            normalized = normalized.replace(from_char, to_char)
        
        # Step 3: Remove combining characters from affricates (t͡s -> ts, t͜s -> ts, etc.)
        # Keep affricates as single phonemes without spaces
        # Remove combining double breve (͡ U+0361) and combining double breve below (͜ U+035C)
        normalized = normalized.replace('͡', '').replace('͜', '')
        
        # Step 4: Remove diacritics not in model
        for diacritic in self.diacritics_to_remove:
            normalized = normalized.replace(diacritic, '')
        
        # Step 5: Remove suprasegmentals not in model
        for suprasegmental in self.suprasegmentals_to_remove:
            normalized = normalized.replace(suprasegmental, '')
        
        # Step 6: Remove characters not in model vocabulary
        for char_to_remove in self.chars_to_remove:
            normalized = normalized.replace(char_to_remove, '')
        
        # Clean up: remove extra spaces and normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def normalize_phoneme_list(self, phonemes: List[str], source: str = 'dictionary') -> List[str]:
        """
        Normalize a list of phonemes.
        
        Args:
            phonemes: List of phoneme strings
            source: Source of phonemes ('dictionary' or 'model')
        
        Returns:
            List of normalized phoneme strings
        """
        normalized_list = []
        for phoneme in phonemes:
            normalized = self.normalize_phoneme_string(phoneme, source=source)
            # If normalized phoneme contains spaces, split it into separate phonemes
            if ' ' in normalized:
                parts = normalized.split()
                normalized_list.extend([p for p in parts if p])  # Add non-empty parts
            elif normalized:  # Only add non-empty phonemes
                normalized_list.append(normalized)
        
        return normalized_list
    
    def normalize_phoneme_char(self, char: str, source: str = 'dictionary') -> str:
        """
        Normalize a single phoneme character.
        
        Args:
            char: Single character to normalize
            source: Source of phoneme ('dictionary' or 'model')
        
        Returns:
            Normalized character
        """
        return self.normalize_phoneme_string(char, source=source)


# Global instance
_phoneme_normalizer = None


def get_phoneme_normalizer(table_path: Optional[Path] = None) -> PhonemeNormalizer:
    """Get or create global phoneme normalizer instance."""
    global _phoneme_normalizer
    if _phoneme_normalizer is None:
        _phoneme_normalizer = PhonemeNormalizer(table_path=table_path)
    return _phoneme_normalizer


def normalize_phonemes(phonemes: List[str], source: str = 'dictionary') -> List[str]:
    """
    Convenience function to normalize phonemes.
    
    Args:
        phonemes: List of phoneme strings
        source: Source of phonemes ('dictionary' or 'model')
    
    Returns:
        List of normalized phoneme strings
    """
    normalizer = get_phoneme_normalizer()
    return normalizer.normalize_phoneme_list(phonemes, source=source)

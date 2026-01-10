"""
G2P module using eSpeak NG for German text-to-phoneme conversion.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    from phonemizer.backend import EspeakBackend
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    EspeakBackend = None


def setup_espeak_library():
    """Setup eSpeak NG library path for macOS."""
    for candidate in [
        Path('/opt/homebrew/lib/libespeak-ng.dylib'),
        Path('/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib'),
        Path('/opt/homebrew/lib/libespeak.dylib'),
        Path('/opt/homebrew/opt/espeak/lib/libespeak.dylib'),
    ]:
        if candidate.exists():
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = str(candidate)
            return str(candidate)
    return None


class G2PConverter:
    """G2P converter using eSpeak NG for German."""
    
    def __init__(self):
        """Initialize G2P converter."""
        self.backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize EspeakBackend."""
        if not HAS_PHONEMIZER:
            print("Warning: phonemizer not installed. Install with: pip install phonemizer")
            return
        
        # Setup library path
        setup_espeak_library()
        
        try:
            self.backend = EspeakBackend(
                language='de',
                punctuation_marks=';:,.!?¡¿—…""''""„"()'
            )
        except RuntimeError as e:
            print(f"Warning: Failed to initialize EspeakBackend: {e}")
            self.backend = None
    
    def get_expected_phonemes(self, text: str) -> List[Dict[str, any]]:
        """
        Get expected phonemes from text using eSpeak NG.
        
        Args:
            text: German text string
            
        Returns:
            List of dictionaries with phoneme information:
            [
                {
                    'phoneme': 'h',
                    'position': 0,  # character position in text
                    'text_char': 'h'  # corresponding character(s)
                },
                ...
            ]
        """
        if self.backend is None:
            return []
        
        try:
            # Get phoneme string
            phoneme_string = self.backend.phonemize([text], strip=True, njobs=1)[0]
            
            # Parse phoneme string and map to text positions
            phonemes = []
            phoneme_chars = phoneme_string.replace(' ', '').replace('ˈ', '').replace('ˌ', '')
            
            # Simple character-to-phoneme mapping (approximate)
            char_idx = 0
            phoneme_idx = 0
            
            while phoneme_idx < len(phoneme_chars) and char_idx < len(text):
                # Skip punctuation and spaces in text
                if text[char_idx].isspace() or not text[char_idx].isalnum():
                    char_idx += 1
                    continue
                
                # Get next phoneme (could be multi-character like 'aɪ̯')
                phoneme = ''
                if phoneme_idx < len(phoneme_chars):
                    phoneme = phoneme_chars[phoneme_idx]
                    phoneme_idx += 1
                    
                    # Check for multi-character phonemes
                    if phoneme_idx < len(phoneme_chars):
                        # Check for common multi-character phonemes
                        two_char = phoneme_chars[phoneme_idx-1:phoneme_idx+1]
                        if two_char in ['aɪ̯', 'aʊ̯', 'ɔʏ̯', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'ɛː', 'aː']:
                            phoneme = two_char
                            phoneme_idx += 1
                
                if phoneme:
                    phonemes.append({
                        'phoneme': phoneme,
                        'position': char_idx,
                        'text_char': text[char_idx] if char_idx < len(text) else '',
                        'phoneme_string': phoneme_string
                    })
                
                char_idx += 1
            
            return phonemes
            
        except Exception as e:
            print(f"Error in G2P conversion: {e}")
            return []
    
    def get_phoneme_string(self, text: str) -> str:
        """
        Get phoneme string from text.
        
        Args:
            text: German text string
            
        Returns:
            Phoneme string in IPA notation
        """
        if self.backend is None:
            return ""
        
        try:
            return self.backend.phonemize([text], strip=True, njobs=1)[0]
        except Exception as e:
            print(f"Error in G2P conversion: {e}")
            return ""


# Global instance
_g2p_converter = None


def get_g2p_converter() -> G2PConverter:
    """Get or create global G2P converter instance."""
    global _g2p_converter
    if _g2p_converter is None:
        _g2p_converter = G2PConverter()
    return _g2p_converter


def get_expected_phonemes(text: str) -> List[Dict[str, any]]:
    """
    Convenience function to get expected phonemes.
    
    Args:
        text: German text string
        
    Returns:
        List of phoneme dictionaries
    """
    converter = get_g2p_converter()
    return converter.get_expected_phonemes(text)



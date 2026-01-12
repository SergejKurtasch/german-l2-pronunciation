"""
G2P module using eSpeak NG for German text-to-phoneme conversion.
"""

import os
import urllib.request
import zipfile
import io
import pickle
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
import config

try:
    from phonemizer.backend import EspeakBackend
    HAS_PHONEMIZER = True
except ImportError:
    HAS_PHONEMIZER = False
    EspeakBackend = None

# Import phoneme normalizer
try:
    from modules.phoneme_normalizer import get_phoneme_normalizer
    HAS_PHONEME_NORMALIZER = True
except ImportError:
    HAS_PHONEME_NORMALIZER = False
    get_phoneme_normalizer = None


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


class DSLG2P:
    """G2P lookup using IPA-Dict-DSL format (DSL dictionary)."""
    
    def __init__(self, dsl_path: Path, download_url: Optional[str] = None):
        """
        Initialize DSL lexicon.
        
        Args:
            dsl_path: Path to DSL file (.dsl)
            download_url: Optional URL to download DSL if not found
        """
        self.dsl_path = dsl_path
        self.lexicon: Dict[str, List[str]] = {}
        
        # Ensure directory exists
        self.dsl_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download if missing
        if not self.dsl_path.exists() and download_url:
            self._download_dsl(download_url)
            
        # Load lexicon
        if self.dsl_path.exists():
            self._load_dsl()
        else:
            print(f"Warning: DSL file not found at {self.dsl_path}")

    def _get_cache_path(self) -> Path:
        """Get path to pickle cache file."""
        return self.dsl_path.with_suffix('.dsl.pickle')

    def _load_from_cache(self) -> bool:
        """Load lexicon from pickle cache if available and up-to-date."""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return False
        
        # Check if cache is newer than DSL file
        try:
            cache_mtime = cache_path.stat().st_mtime
            dsl_mtime = self.dsl_path.stat().st_mtime
            if cache_mtime < dsl_mtime:
                # DSL file is newer, cache is outdated
                return False
        except OSError:
            return False
        
        # Load from cache
        try:
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.lexicon = pickle.load(f)
            load_time = time.time() - start_time
            count = len(self.lexicon)
            print(f"Loaded {count} words from DSL lexicon cache in {load_time:.2f} seconds.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
            return False

    def _save_cache(self):
        """Save lexicon to pickle cache for fast loading."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.lexicon, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _download_dsl(self, url: str):
        """Download DSL file from URL."""
        print(f"Downloading DSL lexicon from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(self.dsl_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading DSL lexicon: {e}")

    def _normalize_ipa_transcription(self, transcription: str) -> List[str]:
        """
        Normalize IPA transcription from DSL format.
        Removes: /, ˌ, ˈ, (), [], and splits into individual phonemes.
        Then applies phoneme normalization from phoneme_normalization_table.json.
        
        Args:
            transcription: Raw IPA string like "/kɔmˈpjuːtɐ/" or "kɔmˈpjuːtɐ"
            
        Returns:
            List of normalized phoneme strings
        """
        import re
        import unicodedata
        
        # #region agent log
        import json, time
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
            debug_f.write(json.dumps({"location":"g2p_module.py:_normalize_ipa_transcription","message":"Input transcription","data":{"transcription":transcription,"has_t_h":'tʰ' in transcription or 't ʰ' in transcription},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
        # #endregion
        
        # Remove leading/trailing slashes and whitespace
        cleaned = transcription.strip().strip('/').strip()
        
        # Remove stress marks (these will also be removed by normalizer, but remove early for parsing)
        cleaned = cleaned.replace('ˈ', '').replace('ˌ', '')
        
        # Remove parentheses and their contents (optional parts)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Remove brackets and their contents
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        
        # Remove word boundaries and other special marks
        cleaned = cleaned.replace('|', '').replace('‖', '')
        
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
            debug_f.write(json.dumps({"location":"g2p_module.py:_normalize_ipa_transcription","message":"After cleaning","data":{"cleaned":cleaned,"has_t_h":'tʰ' in cleaned or 't ʰ' in cleaned},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
        # #endregion
        
        # Split into phonemes (handle multi-character phonemes and combining characters)
        # This is similar to the parsing logic in _parse_phonemes_from_string
        phonemes = []
        i = 0
        while i < len(cleaned):
            # Skip spaces
            if cleaned[i].isspace():
                i += 1
                continue
            
            # Check for multi-character phonemes (diphthongs, affricates, long vowels)
            matched = False
            
            # 3-character phonemes (diphthongs with combining character)
            if i + 3 <= len(cleaned):
                three_char = cleaned[i:i+3]
                if three_char in ['aɪ̯', 'aʊ̯', 'ɔʏ̯']:
                    phonemes.append(three_char)
                    i += 3
                    matched = True
            
            # 2-character phonemes (WITHOUT affricates - they should be separate)
            if not matched and i + 2 <= len(cleaned):
                two_char = cleaned[i:i+2]
                # Long vowels, diphthongs, aspirated consonants ONLY
                # REMOVED affricates: pf, ts, tʃ, dʒ, d͡ʒ, t͜s, d͜ʒ
                if two_char in ['aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'ɛː',
                               'aɪ', 'aʊ', 'ɔʏ',
                               'tʰ', 'pʰ', 'kʰ']:
                    phonemes.append(two_char)
                    i += 2
                    matched = True
            
            # Check for combining characters (modifier letters like ʰ, ʲ, etc.)
            # These should attach to the previous base character
            if not matched and i < len(cleaned):
                char = cleaned[i]
                # Check if current character is a combining character or modifier letter
                # Combining characters: U+0300-U+036F (combining diacritics)
                # Modifier letters: U+02B0-U+02FF (like ʰ U+02B0, ʲ U+02B7)
                char_cat = unicodedata.category(char)
                is_combining = char_cat == 'Mn'  # Mark, nonspacing (combining diacritics)
                is_modifier = '\u02B0' <= char <= '\u02FF'  # Modifier letters
                
                if (is_combining or is_modifier) and phonemes:
                    # Attach to previous phoneme
                    phonemes[-1] = phonemes[-1] + char
                    i += 1
                    matched = True
                elif not char.isspace() and char not in ['/', '(', ')', '[', ']']:
                    # Single character phoneme (base character)
                    phonemes.append(char)
                    i += 1
                    matched = True
                
                if not matched:
                    i += 1
        
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
            debug_f.write(json.dumps({"location":"g2p_module.py:_normalize_ipa_transcription","message":"Phonemes before normalization","data":{"phonemes":phonemes,"has_t_h":any('tʰ' in ph or 't ʰ' in ph for ph in phonemes)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
        # #endregion
        
        # Apply phoneme normalization from phoneme_normalization_table.json
        if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer:
            try:
                normalizer = get_phoneme_normalizer()
                # Normalize each phoneme (source='dictionary' for DSL)
                normalized_phonemes = normalizer.normalize_phoneme_list(phonemes, source='dictionary')
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                    debug_f.write(json.dumps({"location":"g2p_module.py:_normalize_ipa_transcription","message":"Phonemes after normalization","data":{"normalized_phonemes":normalized_phonemes,"has_t_h":any('tʰ' in ph or 't ʰ' in ph for ph in normalized_phonemes)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                # #endregion
                return normalized_phonemes
            except Exception as e:
                print(f"Warning: Failed to normalize phonemes: {e}")
                return phonemes
        
        return phonemes

    def _load_dsl(self):
        """Load DSL lexicon into memory, using cache if available."""
        import re
        import json
        
        # Try to load from cache first
        if self._load_from_cache():
            return
        
        # Cache miss or outdated - load from DSL file
        print(f"Loading DSL lexicon from {self.dsl_path}...")
        start_time = time.time()
        count = 0
        try:
            with open(self.dsl_path, 'r', encoding='utf-8') as f:
                current_word = None
                for line in f:
                    line = line.rstrip('\n\r')
                    
                    # If line without indentation - it's a new word
                    if line and not line.startswith(' ') and not line.startswith('\t'):
                        # Remove tags and formatting
                        word = line.strip()
                        word = re.sub(r'\[.*?\]', '', word).strip()
                        if word:
                            current_word = word.lower()
                            self.lexicon[current_word] = []
                    # If line with indentation - it's a transcription
                    elif line and (line.startswith(' ') or line.startswith('\t')) and current_word:
                        # Extract IPA from [m1]...[/m] tags
                        match = re.search(r'\[m\d*\](.*?)\[/m\]', line)
                        if match:
                            raw_transcription = match.group(1).strip()
                            if raw_transcription:
                                # IMPROVED LOGIC: Normalize FIRST, then check if result is valid
                                # This allows us to load words that contain removable characters
                                # that become valid after normalization
                                
                                # Normalize and split into phonemes
                                phonemes = self._normalize_ipa_transcription(raw_transcription)
                                
                                # Check if normalized result is valid (non-empty, all phonemes are valid)
                                if phonemes and all(ph and ph.strip() and ph != "~" for ph in phonemes):
                                    # Store first transcription (usually the most common)
                                    if not self.lexicon[current_word]:
                                        self.lexicon[current_word] = phonemes
                                        count += 1
                                # If normalization resulted in empty/invalid phonemes, skip this word
                                # (it will use MFA or eSpeak instead)
            
            load_time = time.time() - start_time
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                debug_f.write(json.dumps({"location":"g2p_module.py:dsl_load","message":"DSL lexicon loaded","data":{"count":count,"sample_words":list(self.lexicon.keys())[:5]},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C"})+'\n')
            
            print(f"Loaded {count} words from DSL lexicon in {load_time:.2f} seconds.")
            
            # Save to cache for next time
            self._save_cache()
        except Exception as e:
            import json
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                debug_f.write(json.dumps({"location":"g2p_module.py:dsl_load_error","message":"DSL lexicon load error","data":{"error":str(e)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C"})+'\n')
            print(f"Error loading DSL lexicon: {e}")

    def lookup(self, word: str) -> Optional[List[str]]:
        """
        Lookup word in DSL lexicon.
        
        Args:
            word: Word to look up
            
        Returns:
            List of phonemes if found, None otherwise
        """
        clean_word = word.lower().strip(".,!?;:()\"")
        result = self.lexicon.get(clean_word)
        return result


class LexiconG2P:
    """G2P lookup using a pre-defined lexicon (dictionary) file."""
    
    def __init__(self, lexicon_path: Path, download_url: Optional[str] = None):
        """
        Initialize lexicon.
        
        Args:
            lexicon_path: Path to lexicon file (.dict)
            download_url: Optional URL to download lexicon if not found
        """
        self.lexicon_path = lexicon_path
        self.lexicon: Dict[str, List[str]] = {}
        
        # Ensure directory exists
        self.lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download if missing
        if not self.lexicon_path.exists() and download_url:
            self._download_lexicon(download_url)
            
        # Load lexicon
        if self.lexicon_path.exists():
            self._load_lexicon()
        else:
            print(f"Warning: Lexicon file not found at {self.lexicon_path}")
    
    def _get_cache_path(self) -> Path:
        """Get path to pickle cache file."""
        return self.lexicon_path.with_suffix('.dict.pickle')
    
    def _load_from_cache(self) -> bool:
        """Load lexicon from pickle cache if available and up-to-date."""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return False
        
        # Check if cache is newer than lexicon file
        try:
            cache_mtime = cache_path.stat().st_mtime
            lexicon_mtime = self.lexicon_path.stat().st_mtime
            if cache_mtime < lexicon_mtime:
                # Lexicon file is newer, cache is outdated
                return False
        except OSError:
            return False
        
        # Load from cache
        try:
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                self.lexicon = pickle.load(f)
            load_time = time.time() - start_time
            count = len(self.lexicon)
            print(f"Loaded {count} words from lexicon cache in {load_time:.2f} seconds.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
            return False
    
    def _save_cache(self):
        """Save lexicon to pickle cache for fast loading."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.lexicon, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved lexicon cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def _download_lexicon(self, url: str):
        """Download and extract lexicon from URL."""
        print(f"Downloading lexicon from {url}...")
        try:
            # Use a User-Agent to avoid being blocked by some servers (like GitHub)
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
            )
            
            if url.endswith('.zip'):
                with urllib.request.urlopen(req) as response:
                    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                        # Find the .dict file in the zip
                        dict_files = [f for f in z.namelist() if f.endswith('.dict')]
                        if dict_files:
                            # Extract the first .dict file to our target path
                            with z.open(dict_files[0]) as source, open(self.lexicon_path, 'wb') as target:
                                target.write(source.read())
                            print(f"Extracted {dict_files[0]} to {self.lexicon_path}")
                        else:
                            print("Error: No .dict file found in the zip archive.")
            else:
                with urllib.request.urlopen(req) as response, open(self.lexicon_path, 'wb') as out_file:
                    out_file.write(response.read())
            print("Download and extraction complete.")
        except Exception as e:
            print(f"Error downloading/extracting lexicon: {e}")

    def _load_lexicon(self):
        """Load lexicon into memory, using cache if available."""
        # #region agent log
        import json, time
        # #endregion
        
        # Try to load from cache first
        if self._load_from_cache():
            return
        
        # Cache miss or outdated - load from lexicon file
        print(f"Loading lexicon from {self.lexicon_path}...")
        start_time = time.time()
        count = 0
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        # MFA dictionaries sometimes have probability numbers (like 0.99, 1.0)
                        # We need to filter them out. They are usually floats.
                        phonemes = []
                        for part in parts[1:]:
                            try:
                                # Try to convert to float - if it succeeds, it's a probability, skip it
                                float(part)
                                continue
                            except ValueError:
                                # Not a number, so it's a phoneme
                                phonemes.append(part)
                        
                        if phonemes:
                            # Apply normalization from phoneme_normalization_table.json
                            if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer:
                                try:
                                    normalizer = get_phoneme_normalizer()
                                    # Normalize as dictionary source
                                    normalized_phonemes = normalizer.normalize_phoneme_list(phonemes, source='dictionary')
                                    self.lexicon[word] = normalized_phonemes
                                except Exception as e:
                                    print(f"Warning: Failed to normalize phonemes for word '{word}': {e}")
                                    self.lexicon[word] = phonemes
                            else:
                                self.lexicon[word] = phonemes
                            count += 1
            load_time = time.time() - start_time
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                debug_f.write(json.dumps({"location":"g2p_module.py:load","message":"Lexicon loaded","data":{"count":count,"sample_words":list(self.lexicon.keys())[:5]},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C"})+'\n')
            # #endregion
            print(f"Loaded {count} words from lexicon in {load_time:.2f} seconds.")
            
            # Save to cache for next time
            self._save_cache()
        except Exception as e:
            # #region agent log
            with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                debug_f.write(json.dumps({"location":"g2p_module.py:load_error","message":"Lexicon load error","data":{"error":str(e)},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"C"})+'\n')
            # #endregion
            print(f"Error loading lexicon: {e}")

    def lookup(self, word: str) -> Optional[List[str]]:
        """
        Lookup word in lexicon.
        
        Args:
            word: Word to look up
            
        Returns:
            List of phonemes if found, None otherwise
        """
        # Normalize word (lowercase, remove some punctuation)
        clean_word = word.lower().strip(".,!?;:()\"")
        result = self.lexicon.get(clean_word)
        # #region agent log
        import json, time
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
            debug_f.write(json.dumps({"location":"g2p_module.py:lookup","message":"Lexicon lookup","data":{"word":word,"clean_word":clean_word,"found":result is not None,"phonemes":result},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"A"})+'\n')
        # #endregion
        return result


class G2PConverter:
    """G2P converter using a hybrid approach: DSL (primary) → MFA (fallback) → eSpeak NG (last resort)."""
    
    def __init__(self, load_dicts_immediately: bool = True):
        """
        Initialize G2P converter.
        
        Args:
            load_dicts_immediately: If True, load dictionaries immediately. 
                                   If False, load them lazily on first use.
        """
        self.backend = None
        self.dsl_lexicon = None
        self.mfa_lexicon = None
        self._dicts_loaded = False
        self._load_dicts_immediately = load_dicts_immediately
        
        if load_dicts_immediately:
            self._load_dictionaries()
        else:
            # Initialize backend only (lightweight)
            self._initialize_backend()
    
    def _load_dictionaries(self):
        """Load dictionaries (DSL and MFA) in parallel."""
        if self._dicts_loaded:
            return
        
        # #region agent log
        import json
        dict_load_start = time.time()
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"g2p_module.py:_load_dictionaries:start","message":"Starting dictionary loading","data":{},"timestamp":int(time.time()*1000),"elapsed_ms":0})+'\n')
        # #endregion
        
        # Load dictionaries in parallel threads
        dsl_lexicon = [None]
        mfa_lexicon = [None]
        dsl_error = [None]
        mfa_error = [None]
        dsl_time = [0]
        mfa_time = [0]
        
        def load_dsl():
            dsl_start = time.time()
            try:
                dsl_lexicon[0] = DSLG2P(
                    dsl_path=config.IPA_DSL_LEXICON_PATH,
                    download_url=config.IPA_DSL_LEXICON_URL
                )
            except Exception as e:
                dsl_error[0] = e
            finally:
                dsl_time[0] = (time.time() - dsl_start) * 1000
        
        def load_mfa():
            mfa_start = time.time()
            try:
                mfa_lexicon[0] = LexiconG2P(
                    lexicon_path=config.MFA_GERMAN_LEXICON_PATH,
                    download_url=config.MFA_GERMAN_LEXICON_URL
                )
            except Exception as e:
                mfa_error[0] = e
            finally:
                mfa_time[0] = (time.time() - mfa_start) * 1000
        
        # Start both threads
        dsl_thread = threading.Thread(target=load_dsl, daemon=False)
        mfa_thread = threading.Thread(target=load_mfa, daemon=False)
        
        dsl_thread.start()
        mfa_thread.start()
        
        # Wait for both to complete
        dsl_thread.join()
        mfa_thread.join()
        
        dict_load_elapsed = (time.time() - dict_load_start) * 1000
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"g2p_module.py:_load_dictionaries:end","message":"Dictionary loading completed","data":{"dsl_time_ms":dsl_time[0],"mfa_time_ms":mfa_time[0],"total_time_ms":dict_load_elapsed},"timestamp":int(time.time()*1000),"elapsed_ms":int(dict_load_elapsed)})+'\n')
        # #endregion
        
        # Assign results
        if dsl_error[0]:
            print(f"Warning: Failed to load DSL lexicon: {dsl_error[0]}")
        else:
            self.dsl_lexicon = dsl_lexicon[0]
        
        if mfa_error[0]:
            print(f"Warning: Failed to load MFA lexicon: {mfa_error[0]}")
        else:
            self.mfa_lexicon = mfa_lexicon[0]
        
        self._dicts_loaded = True
    
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
    
    def _normalize_phoneme_char(self, char: str) -> str:
        """
        Normalize phoneme character using phoneme_normalization_table.json.
        
        This method now uses the unified normalization table instead of custom mapping.
        Only applies Unicode normalization (g -> ɡ), does NOT merge different phonemes.
        
        Args:
            char: Character to normalize
            
        Returns:
            Normalized character
        """
        if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer:
            try:
                normalizer = get_phoneme_normalizer()
                # Normalize as dictionary source (removes diacritics not in model, etc.)
                normalized = normalizer.normalize_phoneme_char(char, source='dictionary')
                return normalized
            except Exception as e:
                print(f"Warning: Failed to normalize phoneme character: {e}")
                return char
        
        # Fallback: return as-is if normalizer not available
        return char
    
    def _parse_phonemes_from_string(self, phoneme_string: str) -> List[str]:
        """
        Parse phonemes from eSpeak output string, correctly handling multi-character phonemes.
        Handles cases where eSpeak may insert spaces between characters of the same phoneme.
        
        Args:
            phoneme_string: Raw phoneme string from eSpeak (may have spaces between characters)
            
        Returns:
            List of phoneme strings (each phoneme is correctly grouped)
        """
        # Remove stress marks but preserve combining characters for now
        cleaned = phoneme_string.replace('ˈ', '').replace('ˌ', '')
        
        # All German multi-character phonemes (diphthongs, long vowels, aspirated)
        # Note: eSpeak may output these with or without combining characters (̯)
        # REMOVED affricates - they should be separate phonemes: pf, ts, tʃ, dʒ
        multi_char_phonemes = [
            # Diphthongs (with and without combining character)
            'aɪ', 'aɪ̯', 'aʊ', 'aʊ̯', 'ɔʏ', 'ɔʏ̯',
            # Long vowels
            'aː', 'eː', 'iː', 'oː', 'uː', 'yː', 'øː', 'ɛː',
            # Aspirated consonants (with modifier letter ʰ)
            'tʰ', 'pʰ', 'kʰ',
        ]
        
        # German vowels that can be long (for length mark attachment)
        vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'ø', 'ɛ', 'ɑ', 'ɜ']
        
        # First, remove all spaces to get continuous string
        # This handles cases where eSpeak outputs "a ʊ" instead of "aʊ"
        no_spaces = cleaned.replace(' ', '')
        
        phonemes = []
        i = 0
        
        while i < len(no_spaces):
            matched = False
            
            # Check 3-character phonemes first (diphthongs with combining character)
            if i + 3 <= len(no_spaces):
                three_char = no_spaces[i:i+3]
                if three_char in ['aɪ̯', 'aʊ̯', 'ɔʏ̯']:
                    phonemes.append(three_char)
                    i += 3
                    matched = True
            
            # Check 2-character phonemes
            if not matched and i + 2 <= len(no_spaces):
                two_char = no_spaces[i:i+2]
                # Check both with and without combining character
                two_char_no_combining = two_char.replace('̯', '')
                
                # Check if it matches any multi-character phoneme
                if two_char in multi_char_phonemes or two_char_no_combining in multi_char_phonemes:
                    # Use version without combining character for consistency
                    if '̯' in two_char:
                        phonemes.append(two_char_no_combining)
                    else:
                        phonemes.append(two_char)
                    i += 2
                    matched = True
            
            # Single character phoneme
            if not matched:
                char = no_spaces[i]
                # Skip combining characters that weren't part of a match
                if char == '̯':
                    i += 1
                    continue
                
                # Check if next character is a modifier letter (like ʰ, ʲ) or combining character
                # These should attach to the current base character
                import unicodedata
                if i + 1 < len(no_spaces):
                    next_char = no_spaces[i + 1]
                    next_char_cat = unicodedata.category(next_char)
                    is_combining = next_char_cat == 'Mn'  # Mark, nonspacing
                    is_modifier = '\u02B0' <= next_char <= '\u02FF'  # Modifier letters
                    
                    if is_combining or is_modifier:
                        # Attach modifier/combining character to base character
                        phonemes.append(char + next_char)
                        i += 2
                        matched = True
                
                # Check if next character is length mark (ː) - attach it to vowel
                if not matched and i + 1 < len(no_spaces) and no_spaces[i + 1] == 'ː':
                    # Check if current char is a vowel
                    if char in vowels:
                        phonemes.append(char + 'ː')
                        i += 2  # Skip both the vowel and the length mark
                    else:
                        # Not a vowel - add separately (length mark shouldn't attach to consonants)
                        phonemes.append(char)
                        i += 1
                elif not matched:
                    phonemes.append(char)
                    i += 1
        
        return phonemes
    
    def get_expected_phonemes(self, text: str) -> List[Dict[str, any]]:
        """
        Get expected phonemes from text using priority chain: DSL → MFA → eSpeak NG.
        
        Args:
            text: German text string
            
        Returns:
            List of dictionaries with phoneme information
        """
        # Lazy load dictionaries if not loaded yet
        if not self._dicts_loaded:
            self._load_dictionaries()
        
        if self.backend is None and self.dsl_lexicon is None and self.mfa_lexicon is None:
            return []
        
        # Split text into words while preserving punctuation positions for mapping
        import re
        # This regex finds words and non-words (spaces, punctuation)
        tokens = re.findall(r"[\w']+|[^\w\s]", text)
        
        # #region agent log
        import json, time
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
            debug_f.write(json.dumps({"location":"g2p_module.py:tokens","message":"Tokens identified","data":{"tokens":tokens},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"D"})+'\n')
        # #endregion

        all_expected_phonemes = []
        current_char_pos = 0
        word_count = 0  # Track number of words processed
        
        # #region agent log
        token_processing_start = time.time()
        # #endregion
        
        for token in tokens:
            # Find token position in original text to keep char_pos accurate
            token_pos = text.find(token, current_char_pos)
            if token_pos == -1:
                token_pos = current_char_pos
            
            # If it's punctuation or space, just skip but update position
            if not token[0].isalnum():
                current_char_pos = token_pos + len(token)
                continue
            
            phonemes = None
            source = None
            dsl_phonemes = None
            
            # Priority 1: Try DSL Lexicon (IPA-Dict-DSL) - best for loanwords
            if self.dsl_lexicon:
                dsl_phonemes = self.dsl_lexicon.lookup(token)
                if dsl_phonemes:
                    # IMPROVED LOGIC: Normalize FIRST, then check if result is valid
                    # This allows us to use DSL words that contain removable characters
                    # that become valid after normalization
                    if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer:
                        try:
                            normalizer = get_phoneme_normalizer()
                            # Normalize DSL phonemes first
                            normalized_dsl_phonemes = normalizer.normalize_phoneme_list(dsl_phonemes, source='dictionary')
                            
                            # Check if normalized result is valid (non-empty, all phonemes are valid)
                            if normalized_dsl_phonemes and all(ph.strip() and ph != "~" for ph in normalized_dsl_phonemes):
                                # Use normalized DSL phonemes
                                phonemes = normalized_dsl_phonemes
                                source = 'dsl'
                                # #region agent log
                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                                    debug_f.write(json.dumps({"location":"g2p_module.py:process_token","message":"Using DSL lexicon (normalized)","data":{"token":token,"raw_phonemes":dsl_phonemes,"normalized_phonemes":phonemes},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                                # #endregion
                            else:
                                # Normalization resulted in empty/invalid phonemes - skip DSL
                                # #region agent log
                                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                                    debug_f.write(json.dumps({"location":"g2p_module.py:process_token","message":"DSL phonemes invalid after normalization, skipping DSL","data":{"token":token,"raw_phonemes":dsl_phonemes,"normalized_phonemes":normalized_dsl_phonemes},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                                # #endregion
                                phonemes = None  # Will try next source
                        except Exception as e:
                            print(f"Warning: Failed to normalize DSL phonemes for '{token}': {e}")
                            # Fallback: use DSL phonemes as-is (old behavior)
                            phonemes = dsl_phonemes
                            source = 'dsl'
                    else:
                        # Normalizer not available - use DSL phonemes as-is
                        phonemes = dsl_phonemes
                        source = 'dsl'
                        # #region agent log
                        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                            debug_f.write(json.dumps({"location":"g2p_module.py:process_token","message":"Using DSL lexicon (no normalizer)","data":{"token":token,"phonemes":phonemes},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                        # #endregion
            
            # Priority 2: Try MFA Lexicon (fallback)
            if not phonemes and self.mfa_lexicon:
                mfa_phonemes = self.mfa_lexicon.lookup(token)
                if mfa_phonemes:
                    phonemes = mfa_phonemes
                    source = 'mfa'
                    # #region agent log
                    with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                        debug_f.write(json.dumps({"location":"g2p_module.py:process_token","message":"Using MFA lexicon","data":{"token":token,"phonemes":phonemes},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                    # #endregion
            
            # Priority 3: Fallback to eSpeak NG (last resort)
            if not phonemes and self.backend:
                # #region agent log
                with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as debug_f:
                    debug_f.write(json.dumps({"location":"g2p_module.py:process_token","message":"Using espeak fallback","data":{"token":token},"timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"B"})+'\n')
                # #endregion
                try:
                    # Get phoneme string for this single word
                    phoneme_string = self.backend.phonemize([token], strip=True, njobs=1)[0]
                    phonemes = self._parse_phonemes_from_string(phoneme_string)
                    # Apply normalization to eSpeak phonemes (they come from dictionary-like source)
                    if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer and phonemes:
                        try:
                            normalizer = get_phoneme_normalizer()
                            # Convert ARPAbet to IPA first (eSpeak may output ARPAbet)
                            ipa_phonemes = [config.convert_arpabet_to_ipa(ph) for ph in phonemes]
                            # Normalize as dictionary source
                            phonemes = normalizer.normalize_phoneme_list(ipa_phonemes, source='dictionary')
                        except Exception as e:
                            print(f"Warning: Failed to normalize eSpeak phonemes: {e}")
                    source = 'espeak'
                except Exception as e:
                    print(f"Error in G2P conversion for token '{token}': {e}")
                    phonemes = None
            
            # Process found phonemes
            if phonemes:
                # Apply normalization using phoneme_normalization_table.json
                # Note: DSL phonemes are already normalized above, but MFA and eSpeak need normalization
                if HAS_PHONEME_NORMALIZER and get_phoneme_normalizer:
                    try:
                        normalizer = get_phoneme_normalizer()
                        # Convert ARPAbet to IPA first (if needed, mainly for MFA)
                        ipa_phonemes = [config.convert_arpabet_to_ipa(ph) for ph in phonemes]
                        
                        # Only normalize if not already normalized (DSL is normalized above)
                        if source == 'dsl':
                            # DSL phonemes are already normalized, just convert ARPAbet if needed
                            normalized_phonemes = ipa_phonemes
                        else:
                            # Normalize MFA or eSpeak phonemes
                            normalized_phonemes = normalizer.normalize_phoneme_list(ipa_phonemes, source='dictionary')
                    except Exception as e:
                        print(f"Warning: Failed to normalize phonemes: {e}")
                        # Fallback to old method
                        normalized_phonemes = []
                        for ph in phonemes:
                            ipa_ph = config.convert_arpabet_to_ipa(ph)
                            final_ph = "".join([self._normalize_phoneme_char(c) for c in ipa_ph])
                            normalized_phonemes.append(final_ph)
                else:
                    # Fallback if normalizer not available
                    normalized_phonemes = []
                    for ph in phonemes:
                        ipa_ph = config.convert_arpabet_to_ipa(ph)
                        final_ph = "".join([self._normalize_phoneme_char(c) for c in ipa_ph])
                        normalized_phonemes.append(final_ph)
                
                for ph in normalized_phonemes:
                    all_expected_phonemes.append({
                        'phoneme': ph,
                        'position': token_pos,
                        'text_char': token,
                        'source': source or 'unknown'
                    })
                
                # Word boundary markers removed - no longer adding '||' dividers
            
            current_char_pos = token_pos + len(token)
        
        token_processing_elapsed = (time.time() - token_processing_start) * 1000
        # #region agent log
        with open('/Volumes/SSanDisk/SpeechRec-German-diagnostic/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"performance","hypothesisId":"PERF","location":"g2p_module.py:get_expected_phonemes:token_processing","message":"Token processing completed","data":{"tokens_count":len(tokens),"phonemes_count":len(all_expected_phonemes)},"timestamp":int(time.time()*1000),"elapsed_ms":int(token_processing_elapsed)})+'\n')
        # #endregion
        
        return all_expected_phonemes
    
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


def get_g2p_converter(load_dicts_immediately: bool = False) -> G2PConverter:
    """
    Get or create global G2P converter instance.
    
    Args:
        load_dicts_immediately: If True, load dictionaries immediately. 
                               If False, load them lazily on first use (default).
    """
    global _g2p_converter
    if _g2p_converter is None:
        _g2p_converter = G2PConverter(load_dicts_immediately=load_dicts_immediately)
    return _g2p_converter


def load_g2p_dictionaries():
    """Force load G2P dictionaries (for background loading)."""
    converter = get_g2p_converter(load_dicts_immediately=False)
    if not converter._dicts_loaded:
        print("Loading G2P dictionaries in background...")
        converter._load_dictionaries()
        print("G2P dictionaries loaded successfully!")


# Cache for G2P results to speed up repeated lookups
_g2p_cache = {}
_g2p_cache_max_size = 1000  # Limit cache size

def get_expected_phonemes(text: str) -> List[Dict[str, any]]:
    """
    Convenience function to get expected phonemes.
    Uses caching to speed up repeated lookups.
    
    Args:
        text: German text string
        
    Returns:
        List of phoneme dictionaries
    """
    # Check cache first
    if text in _g2p_cache:
        return _g2p_cache[text]
    
    # Get phonemes from converter
    converter = get_g2p_converter()
    result = converter.get_expected_phonemes(text)
    
    # Cache result (with size limit)
    if len(_g2p_cache) >= _g2p_cache_max_size:
        # Remove oldest entry (simple FIFO)
        _g2p_cache.pop(next(iter(_g2p_cache)))
    _g2p_cache[text] = result
    
    return result



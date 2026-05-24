"""
Wav2Vec2Phoneme recognition module.
This module provides phoneme recognition using Wav2Vec2Phoneme models.
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers - check if library is available
# Use a function to check at runtime, not at import time
def _check_transformers():
    """Check if transformers is available."""
    try:
        import transformers
        return True, transformers
    except ImportError:
        return False, None

HAS_TRANSFORMERS, transformers = _check_transformers()

# Try to import specific classes (may not all be available)
Wav2Vec2PhonemeForCTC = None
Wav2Vec2PhonemeCTCTokenizer = None
Wav2Vec2FeatureExtractor = None
AutoModelForCTC = None
AutoTokenizer = None
AutoProcessor = None
Wav2Vec2ForCTC = None
Wav2Vec2Processor = None
Wav2Vec2CTCTokenizer = None

# Import specific classes dynamically when needed
# We'll import them in the _load_model method to avoid import errors at module level


class Wav2Vec2PhonemeRecognizer:
    """Phoneme recognizer using Wav2Vec2Phoneme model.
    
    This model uses Wav2Vec2PhonemeForCTC and Wav2Vec2PhonemeCTCTokenizer
    for phoneme recognition. The tokenizer automatically handles phonemization.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Wav2Vec2Phoneme recognizer.
        
        Args:
            model_name: Hugging Face model name. If None, uses default.
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        # Check transformers availability at runtime
        has_transformers, transformers_module = _check_transformers()
        if not has_transformers:
            raise ImportError(
                f"transformers library is required but could not be imported.\n"
                f"Install with: pip install transformers\n"
                f"Make sure you're using the correct Python environment."
            )
        
        # Default model - try to find a Wav2Vec2Phoneme model
        if model_name is None:
            try:
                import config
                model_name = getattr(config, 'WAV2VEC2_PHONEME_MODEL_NAME', None)
            except:
                pass
            
            # Fallback to a known Wav2Vec2Phoneme model
            if model_name is None:
                # Try common Wav2Vec2Phoneme models
                # Note: We'll use a base model that supports phoneme recognition
                model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        
        self.model_name = model_name
        self.device = device or self._get_device()
        self.tokenizer = None
        self.feature_extractor = None
        self.model = None
        self.vocab = None
        
        self._load_model()
    
    def _get_device(self) -> str:
        """Auto-detect device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _load_model(self):
        """Load Wav2Vec2Phoneme model, tokenizer and feature extractor."""
        try:
            print(f"Loading Wav2Vec2Phoneme model: {self.model_name}")
            
            # Import transformers classes dynamically
            has_transformers, transformers_module = _check_transformers()
            if not has_transformers:
                raise RuntimeError("transformers library is not available")
            
            # Try to import specific classes
            Wav2Vec2PhonemeForCTC_local = None
            Wav2Vec2PhonemeCTCTokenizer_local = None
            Wav2Vec2FeatureExtractor_local = None
            AutoModelForCTC_local = None
            AutoTokenizer_local = None
            AutoProcessor_local = None
            Wav2Vec2ForCTC_local = None
            Wav2Vec2Processor_local = None
            
            try:
                from transformers import Wav2Vec2PhonemeForCTC as Wav2Vec2PhonemeForCTC_local
                from transformers import Wav2Vec2PhonemeCTCTokenizer as Wav2Vec2PhonemeCTCTokenizer_local
            except (ImportError, AttributeError, ModuleNotFoundError):
                pass
            
            try:
                from transformers import Wav2Vec2FeatureExtractor as Wav2Vec2FeatureExtractor_local
            except (ImportError, AttributeError, ModuleNotFoundError):
                pass
            
            try:
                from transformers import AutoModelForCTC as AutoModelForCTC_local
                from transformers import AutoTokenizer as AutoTokenizer_local
                from transformers import AutoProcessor as AutoProcessor_local
            except (ImportError, AttributeError, ModuleNotFoundError):
                pass
            
            try:
                from transformers import Wav2Vec2ForCTC as Wav2Vec2ForCTC_local
                from transformers import Wav2Vec2Processor as Wav2Vec2Processor_local
            except (ImportError, AttributeError, ModuleNotFoundError):
                pass
            
            # Method 1: Try Wav2Vec2Phoneme specific classes
            if Wav2Vec2PhonemeForCTC_local is not None and Wav2Vec2PhonemeCTCTokenizer_local is not None:
                try:
                    print("Trying Wav2Vec2PhonemeForCTC and Wav2Vec2PhonemeCTCTokenizer...")
                    self.tokenizer = Wav2Vec2PhonemeCTCTokenizer_local.from_pretrained(self.model_name)
                    self.feature_extractor = Wav2Vec2FeatureExtractor_local.from_pretrained(self.model_name) if Wav2Vec2FeatureExtractor_local else None
                    self.model = Wav2Vec2PhonemeForCTC_local.from_pretrained(self.model_name)
                    print("Successfully loaded using Wav2Vec2Phoneme classes")
                except Exception as e:
                    print(f"Wav2Vec2Phoneme classes failed: {e}")
                    # Continue to next method
                    pass
            
            # Method 2: Try AutoModelForCTC and AutoProcessor (most universal)
            if self.model is None and AutoModelForCTC_local is not None:
                try:
                    print("Trying AutoModelForCTC and AutoProcessor...")
                    self.model = AutoModelForCTC_local.from_pretrained(self.model_name)
                    try:
                        processor = AutoProcessor_local.from_pretrained(self.model_name)
                        if hasattr(processor, 'tokenizer'):
                            self.tokenizer = processor.tokenizer
                        if hasattr(processor, 'feature_extractor'):
                            self.feature_extractor = processor.feature_extractor
                    except:
                        # Try separate tokenizer and feature extractor
                        try:
                            if AutoTokenizer_local:
                                self.tokenizer = AutoTokenizer_local.from_pretrained(self.model_name)
                        except:
                            pass
                        try:
                            if Wav2Vec2FeatureExtractor_local:
                                self.feature_extractor = Wav2Vec2FeatureExtractor_local.from_pretrained(self.model_name)
                        except:
                            pass
                    print("Successfully loaded using AutoModelForCTC")
                except Exception as e:
                    print(f"AutoModelForCTC failed: {e}")
                    # Continue to next method
                    pass
            
            # Method 3: Try regular Wav2Vec2ForCTC with processor
            if self.model is None and Wav2Vec2ForCTC_local is not None:
                try:
                    print("Trying Wav2Vec2ForCTC with Wav2Vec2Processor...")
                    processor = Wav2Vec2Processor_local.from_pretrained(self.model_name)
                    self.model = Wav2Vec2ForCTC_local.from_pretrained(self.model_name)
                    if hasattr(processor, 'tokenizer'):
                        self.tokenizer = processor.tokenizer
                    if hasattr(processor, 'feature_extractor'):
                        self.feature_extractor = processor.feature_extractor
                    print("Successfully loaded using Wav2Vec2ForCTC")
                except Exception as e:
                    print(f"Wav2Vec2ForCTC failed: {e}")
                    # Continue to next method
                    pass
            
            # If all methods failed, raise error
            if self.model is None:
                error_msg = (
                    f"Failed to load model '{self.model_name}' using any available method.\n"
                    f"Possible solutions:\n"
                    f"1. Check if the model name is correct on https://huggingface.co/models\n"
                    f"2. Try using a different model trained for phoneme recognition\n"
                    f"3. Check model tags: https://huggingface.co/models?other=phoneme-recognition\n"
                    f"4. Try models like: facebook/wav2vec2-base-960h, facebook/wav2vec2-large-960h-lv60-self"
                )
                raise RuntimeError(error_msg)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get vocabulary from tokenizer
            if self.tokenizer:
                if hasattr(self.tokenizer, 'get_vocab'):
                    self.vocab = self.tokenizer.get_vocab()
                elif hasattr(self.tokenizer, 'vocab'):
                    self.vocab = self.tokenizer.vocab
                else:
                    # Build vocab from convert_ids_to_tokens
                    self.vocab = {}
                    vocab_size = getattr(self.tokenizer, 'vocab_size', 1000)
                    for i in range(vocab_size):
                        try:
                            token = self.tokenizer.convert_ids_to_tokens(i)
                            if token is not None:
                                self.vocab[str(token)] = i
                        except (IndexError, KeyError, ValueError):
                            continue
            
            print(f"Wav2Vec2Phoneme model loaded on device: {self.device}")
            if self.vocab:
                print(f"Vocabulary size: {len(self.vocab)}")
            
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load Wav2Vec2Phoneme model {self.model_name}: {e}")
    
    def recognize_phonemes(
        self,
        audio_path: str,
        sample_rate: int = 16000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recognize phonemes from audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate (model expects 16kHz)
            
        Returns:
            Tuple of (logits, emissions):
            - logits: Raw model output (batch, time, vocab_size)
            - emissions: Log-softmax of logits (for forced alignment)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Process audio through feature extractor
        if self.feature_extractor:
            input_values = self.feature_extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
        else:
            # Fallback: use processor if available
            if hasattr(self, 'processor') and self.processor:
                input_values = self.processor(
                    audio,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_values.to(self.device)
            else:
                # Manual processing as last resort
                input_values = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Compute emissions (log-softmax) for forced alignment
        emissions = torch.log_softmax(logits, dim=-1)
        
        return logits, emissions
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary (token to ID mapping).
        
        Returns:
            Dictionary mapping tokens to IDs
        """
        return self.vocab or {}
    
    def decode_phonemes(self, logits: torch.Tensor) -> str:
        """
        Decode logits to phoneme string using greedy CTC decoding.
        Properly handles CTC blank tokens and removes repetitions.
        Uses vocab.json to correctly decode IPA phoneme symbols.
        
        Uses greedy decoding for honest pronunciation diagnosis - reflects actual
        pronunciation without "correcting" user errors.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            
        Returns:
            Decoded phoneme string with IPA symbols
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")
        
        # Determine blank token ID from vocab
        blank_id = 0  # Default CTC blank ID
        if self.vocab:
            blank_tokens = ['|', '<pad>', '<blank>', '[PAD]', '[BLANK]', 'pad', 'blank']
            for blank_token in blank_tokens:
                if blank_token in self.vocab:
                    blank_id = self.vocab[blank_token]
                    break
        
        # Get PAD token ID if available
        pad_token_id = None
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        
        # Use greedy decoding (argmax at each time step)
        predicted_ids = torch.argmax(logits, dim=-1)
        sequence = predicted_ids[0].cpu().tolist()
        
        # CTC collapse: remove blanks and consecutive duplicates
        decoded_ids = []
        prev_id = None
        id_to_token = {v: k for k, v in (self.vocab.items() if self.vocab else {})}
        skip_tokens = {'|', '[PAD]', '<pad>', '<blank>', '[BLANK]', 'h#', 'spn', ''}
        
        for token_id in sequence:
            token_name = id_to_token.get(token_id, '')
            
            # Skip blank tokens, PAD tokens, and special silence tokens
            if (token_id == blank_id or 
                (pad_token_id is not None and token_id == pad_token_id) or
                token_name in skip_tokens):
                prev_id = None
                continue
            # Skip consecutive duplicates (CTC collapse)
            if token_id != prev_id:
                decoded_ids.append(token_id)
                prev_id = token_id
            else:
                prev_id = None  # Reset to allow same token later
        
        # Convert IDs to tokens using vocab.json for proper IPA decoding
        # CRITICAL: Always use vocab.json, never use tokenizer.batch_decode() which returns letters!
        if not self.vocab:
            print("Warning: Vocabulary not loaded for Wav2Vec2Phoneme, cannot decode properly")
            return ""
        
        # Use vocabulary from vocab.json to convert IDs to IPA tokens
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        skip_tokens = {'[PAD]', '[UNK]', '<pad>', '<unk>', '<blank>', '[BLANK]', 
                      '<s>', '</s>', '<|endoftext|>', '|', 'h#', 'spn', ''}
        
        for token_id in decoded_ids:
            token = id_to_token.get(token_id, '')
            # Filter out special tokens but preserve IPA symbols
            if token and token not in skip_tokens:
                tokens.append(token)
        
        # Join tokens with space to preserve IPA symbols
        transcription = ' '.join(tokens)
        
        return transcription
    
# Global instance
_wav2vec2_phoneme_recognizer = None


def get_wav2vec2_phoneme_recognizer(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> Wav2Vec2PhonemeRecognizer:
    """Get or create global Wav2Vec2Phoneme recognizer instance."""
    global _wav2vec2_phoneme_recognizer
    if _wav2vec2_phoneme_recognizer is None:
        _wav2vec2_phoneme_recognizer = Wav2Vec2PhonemeRecognizer(model_name=model_name, device=device)
    return _wav2vec2_phoneme_recognizer


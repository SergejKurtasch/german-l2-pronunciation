"""
Phoneme recognition module using vitouphy/wav2vec2-xls-r-300m-phoneme model.
The model is loaded via Wav2Vec2ForCTC and uses vocab.json for correct IPA phoneme decoding.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
import json
warnings.filterwarnings('ignore')

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Wav2Vec2Processor = None
    Wav2Vec2ForCTC = None


class PhonemeRecognizer:
    """Phoneme recognizer using vitouphy/wav2vec2-xls-r-300m-phoneme model.
    
    The model is loaded via Wav2Vec2ForCTC and the processor automatically loads
    vocab.json from the model repository for correct decoding of IPA phoneme symbols.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize phoneme recognizer.
        
        Args:
            model_name: Hugging Face model name. If None, uses default.
            device: Device to use ('cpu', 'cuda', 'mps'). If None, auto-detect.
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        # Default model (use config default if available)
        if model_name is None:
            try:
                import config
                model_name = config.MODEL_NAME
            except:
                # Fallback to base model if config not available
                model_name = "facebook/wav2vec2-xls-r-300m"
        
        self.model_name = model_name
        self.device = device or self._get_device()
        self.processor = None
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
        """Load Wav2Vec2 model and processor with proper vocab.json handling for IPA phonemes."""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Try to load processor and model
            try:
                # Try loading model first (without processor) to avoid tokenizer issues
                # Some models have tokenizers that require additional dependencies
                try:
                    self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
                except Exception as model_error:
                    # If direct model load fails, try with processor
                    self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                    self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
                
                # Load processor if not already loaded
                if self.processor is None:
                    try:
                        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                    except Exception as proc_error:
                        # If processor fails (e.g., due to tokenizer requiring espeak),
                        # load feature extractor and vocab.json separately
                        from transformers import Wav2Vec2FeatureExtractor
                        from huggingface_hub import hf_hub_download
                        import os
                        
                        # Load feature extractor
                        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
                        
                        # Load vocab.json directly from Hugging Face
                        try:
                            vocab_path = hf_hub_download(repo_id=self.model_name, filename="vocab.json")
                            if vocab_path and os.path.exists(vocab_path):
                                with open(vocab_path, 'r', encoding='utf-8') as f:
                                    vocab_dict = json.load(f)
                            else:
                                vocab_dict = {}
                        except Exception as vocab_error:
                            vocab_dict = {}
                        
                        # Create a minimal processor with only feature extractor
                        # We'll use vocab_dict for decoding instead of tokenizer
                        self.processor = feature_extractor  # Use feature extractor directly
                        self.vocab = vocab_dict  # Store vocab for manual decoding
            except Exception as e:
                # If model doesn't exist or is not a CTC model, provide helpful error
                error_msg = (
                    f"Failed to load model '{self.model_name}'. "
                    f"This model may not exist on Hugging Face or may not be a CTC model.\n"
                    f"Error: {str(e)}\n\n"
                    f"Possible solutions:\n"
                    f"1. Check if the model name is correct on https://huggingface.co/models\n"
                    f"2. If the model is private, authenticate with: huggingface-cli login\n"
                    f"3. Use a different model trained for phoneme recognition\n"
                    f"4. For phoneme recognition, you may need a fine-tuned model or custom tokenizer"
                )
                raise RuntimeError(error_msg)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Get vocabulary (token to ID mapping) from vocab.json
            # If vocab was already loaded directly (from vocab.json), use it
            # Otherwise, try to get it from tokenizer
            if not hasattr(self, 'vocab') or len(self.vocab) == 0:
                self.vocab = {}
            
            # Method 1: Try get_vocab() method (most reliable)
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
                tokenizer = self.processor.tokenizer
                
                # Try get_vocab() first
                if hasattr(tokenizer, 'get_vocab'):
                    self.vocab = tokenizer.get_vocab()
                    print("Loaded vocabulary using tokenizer.get_vocab()")
                
                # Method 2: Try direct vocab attribute
                elif hasattr(tokenizer, 'vocab'):
                    self.vocab = tokenizer.vocab
                    print("Loaded vocabulary using tokenizer.vocab")
                
                # Method 3: Build vocab from convert_ids_to_tokens (for character-level tokenizers)
                elif hasattr(tokenizer, 'convert_ids_to_tokens'):
                    vocab_size = getattr(tokenizer, 'vocab_size', None)
                    if vocab_size is None:
                        # Try to infer vocab size from model config
                        if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                            vocab_size = self.model.config.vocab_size
                        else:
                            vocab_size = 1000  # Fallback
                    
                    print(f"Building vocabulary from tokenizer (vocab_size={vocab_size})")
                    for i in range(vocab_size):
                        try:
                            token = tokenizer.convert_ids_to_tokens(i)
                            if token is not None:
                                # Handle special tokens (they might be returned as objects)
                                if isinstance(token, str):
                                    self.vocab[token] = i
                                else:
                                    # Convert to string if needed
                                    self.vocab[str(token)] = i
                        except (IndexError, KeyError, ValueError):
                            # Skip invalid token IDs
                            continue
                    print("Built vocabulary from tokenizer.convert_ids_to_tokens()")
                
                # Method 4: Try loading vocab.json directly from cache
                if len(self.vocab) == 0:
                    try:
                        from transformers import AutoTokenizer
                        import os
                        from huggingface_hub import cached_assets_path
                        
                        # Try to find vocab.json in cache
                        cache_path = os.path.join(
                            os.path.expanduser("~/.cache/huggingface/hub"),
                            f"models--{self.model_name.replace('/', '--')}"
                        )
                        
                        # Alternative: use tokenizer's saved_vocab_files
                        if hasattr(tokenizer, 'save_vocabulary'):
                            # This indicates vocab.json exists
                            print("Attempting to load vocab.json from model cache...")
                    except Exception:
                        pass
            
            # Verify vocab was loaded (either from tokenizer or directly from vocab.json)
            if len(self.vocab) == 0:
                raise RuntimeError(
                    f"Failed to load vocabulary from model '{self.model_name}'. "
                    f"The vocab.json file may be missing or the tokenizer is not properly configured."
                )
            
            print(f"Model loaded on device: {self.device}")
            print(f"Vocabulary size: {len(self.vocab)}")
            
            # Print sample of IPA phonemes from vocab (for debugging)
            ipa_samples = [token for token in self.vocab.keys() 
                          if any(char in token for char in 'ɪɛɔʊʏœøːɐʁçʃʒŋ') or 
                          len(token) == 1 and token.isalpha()]
            if ipa_samples:
                print(f"Sample IPA phonemes in vocab: {ipa_samples[:20]}")
            
            # Warn if vocabulary is empty or very small (might indicate wrong model type)
            if len(self.vocab) < 20:
                print(f"Warning: Vocabulary size ({len(self.vocab)}) is very small. "
                      f"This model may not be trained for phoneme recognition.")
            
        except RuntimeError:
            # Re-raise RuntimeError with our custom message
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
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
        
        # Process audio through processor (or feature extractor if processor is just feature extractor)
        if hasattr(self.processor, 'feature_extractor'):
            # Full processor with tokenizer
            input_values = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
        else:
            # Just feature extractor (no tokenizer)
            input_values = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values.to(self.device)
        
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
        return self.vocab
    
    def _ctc_beam_search(
        self,
        logits: torch.Tensor,
        beam_width: int = 5,
        length_penalty: float = 0.6
    ) -> List[int]:
        """
        CTC beam search decoding for better accuracy.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            beam_width: Number of beams
            length_penalty: Length penalty factor
            
        Returns:
            List of token IDs (best path)
        """
        import torch.nn.functional as F
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, time, vocab_size)
        
        # Get blank token ID
        blank_id = 0
        if self.vocab:
            blank_tokens = ['|', '<pad>', '<blank>', '[PAD]', '[BLANK]', 'pad', 'blank']
            for blank_token in blank_tokens:
                if blank_token in self.vocab:
                    blank_id = self.vocab[blank_token]
                    break
        
        batch_size, seq_len, vocab_size = log_probs.shape
        log_probs = log_probs[0]  # (time, vocab_size)
        
        # Initialize beams: (score, sequence, last_token)
        beams = [(0.0, [], None)]
        
        for t in range(seq_len):
            new_beams = []
            beam_dict = {}  # Use dict to merge beams with same (sequence, last_token)
            
            for score, sequence, last_token in beams:
                # Get top-k tokens at this time step (more candidates for better accuracy)
                top_k = min(beam_width * 3, vocab_size)
                top_k_probs, top_k_ids = torch.topk(log_probs[t], top_k)
                top_k_probs = top_k_probs.cpu().numpy()
                top_k_ids = top_k_ids.cpu().numpy()
                
                # Always consider blank token (CTC allows staying in same state)
                blank_score = score + log_probs[t, blank_id].item()
                blank_key = (tuple(sequence), last_token)
                if blank_key not in beam_dict or beam_dict[blank_key][0] < blank_score:
                    beam_dict[blank_key] = (blank_score, sequence.copy(), last_token)
                
                for log_prob, token_id in zip(top_k_probs, top_k_ids):
                    token_id = int(token_id.item())
                    new_score = score + log_prob
                    new_sequence = sequence.copy()
                    new_last_token = last_token
                    
                    # CTC rules:
                    # 1. Skip blank tokens (already handled above)
                    if token_id == blank_id:
                        continue
                    
                    # 2. If same as last token, extend current beam without adding (CTC collapse)
                    if token_id == last_token:
                        # This extends the current sequence but doesn't add new token
                        # The score improves but sequence stays same
                        same_key = (tuple(sequence), last_token)
                        if same_key not in beam_dict or beam_dict[same_key][0] < new_score:
                            beam_dict[same_key] = (new_score, sequence.copy(), last_token)
                        continue
                    
                    # 3. Add new token
                    new_sequence.append(token_id)
                    new_last_token = token_id
                    new_key = (tuple(new_sequence), new_last_token)
                    if new_key not in beam_dict or beam_dict[new_key][0] < new_score:
                        beam_dict[new_key] = (new_score, new_sequence, new_last_token)
            
            # Convert dict to list and keep top beam_width beams
            beams = list(beam_dict.values())
            beams.sort(key=lambda x: x[0], reverse=True)
            beams = beams[:beam_width]
            
            # If no beams, break early
            if not beams:
                break
        
        # Apply length penalty and select best beam
        if beams:
            best_score, best_sequence, _ = max(beams, key=lambda x: x[0] + length_penalty * len(x[1]))
            return best_sequence
        
        return []
    
    def decode_phonemes(self, logits: torch.Tensor, use_beam_search: Optional[bool] = None) -> str:
        """
        Decode logits to phoneme string using CTC decoding (greedy or beam search).
        Properly handles CTC blank tokens and removes repetitions.
        Uses vocab.json to correctly decode IPA phoneme symbols.
        
        Args:
            logits: Model logits (batch, time, vocab_size)
            use_beam_search: Whether to use beam search (None = use config setting)
            
        Returns:
            Decoded phoneme string with IPA symbols
        """
        # Check if beam search should be used
        if use_beam_search is None:
            try:
                import config
                use_beam_search = getattr(config, 'BEAM_SEARCH_ENABLED', False)
                beam_width = getattr(config, 'BEAM_WIDTH', 5)
                length_penalty = getattr(config, 'BEAM_SEARCH_LENGTH_PENALTY', 0.6)
            except:
                use_beam_search = False
                beam_width = 5
                length_penalty = 0.6
        else:
            try:
                import config
                beam_width = getattr(config, 'BEAM_WIDTH', 5)
                length_penalty = getattr(config, 'BEAM_SEARCH_LENGTH_PENALTY', 0.6)
            except:
                beam_width = 5
                length_penalty = 0.6
        # Determine blank token ID from vocab.json
        # For CTC models, blank is typically token with ID 0 or token "|" (pipe symbol)
        blank_id = 0  # Default CTC blank ID
        if self.vocab:
            # Try to find blank token in vocab (CTC uses "|" as blank, ID 0)
            blank_tokens = ['|', '<pad>', '<blank>', '[PAD]', '[BLANK]', 'pad', 'blank']
            for blank_token in blank_tokens:
                if blank_token in self.vocab:
                    blank_id = self.vocab[blank_token]
                    break
        
        # Get PAD token ID if available
        pad_token_id = None
        if hasattr(self.processor, 'tokenizer'):
            if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                pad_token_id = self.processor.tokenizer.pad_token_id
            elif hasattr(self.processor.tokenizer, '_pad_token') and hasattr(self.processor.tokenizer._pad_token, 'id'):
                pad_token_id = self.processor.tokenizer._pad_token.id
        
        # Choose decoding method
        if use_beam_search:
            # Use beam search decoding (already handles CTC collapse)
            decoded_ids = self._ctc_beam_search(logits, beam_width, length_penalty)
            if not decoded_ids:
                # Fallback to greedy if beam search failed
                use_beam_search = False
        
        if not use_beam_search:
            # Use greedy decoding (original method)
            predicted_ids = torch.argmax(logits, dim=-1)  # (batch, time)
            sequence = predicted_ids[0].cpu().tolist()
            
            # CTC collapse: remove blanks and consecutive duplicates
            decoded_ids = []
            prev_id = None
            id_to_token = {v: k for k, v in (self.vocab.items() if self.vocab else {})}
            # Special tokens to skip (silence, padding, etc.)
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
        if not self.vocab:
            # Fallback to processor decode (may not preserve IPA symbols correctly)
            print("Warning: Vocabulary not loaded, using processor.decode() which may not preserve IPA symbols")
            # Get predicted IDs for fallback
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
        else:
            # Use vocabulary from vocab.json to convert IDs to IPA tokens
            # This ensures proper decoding of IPA phoneme symbols
            id_to_token = {v: k for k, v in self.vocab.items()}
            tokens = []
            for token_id in decoded_ids:
                token = id_to_token.get(token_id, '')
                # Filter out special tokens but preserve IPA symbols
                if token:
                    # Skip common special tokens
                    special_tokens = ['[PAD]', '[UNK]', '<pad>', '<unk>', '<blank>', '[BLANK]', 
                                     '<s>', '</s>', '<|endoftext|>']
                    if token not in special_tokens:
                        tokens.append(token)
            
            # Join tokens with space to preserve IPA symbols
            transcription = ' '.join(tokens)
        
        return transcription


# Global instances (one per model)
_phoneme_recognizers: Dict[str, PhonemeRecognizer] = {}


def get_phoneme_recognizer(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> PhonemeRecognizer:
    """Get or create phoneme recognizer instance for a specific model.
    
    Creates separate instances for different models to allow multiple models
    to be loaded simultaneously.
    """
    global _phoneme_recognizers
    
    # Default model (use config default if available)
    if model_name is None:
        try:
            import config
            model_name = config.MODEL_NAME
        except:
            # Fallback to base model if config not available
            model_name = "facebook/wav2vec2-xls-r-300m"
    
    # Use model_name as key to allow multiple models
    if model_name not in _phoneme_recognizers:
        _phoneme_recognizers[model_name] = PhonemeRecognizer(model_name=model_name, device=device)
    
    return _phoneme_recognizers[model_name]


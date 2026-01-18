# German Pronunciation Diagnostic App (L2-Trainer)

A comprehensive diagnostic application for learning German pronunciation that compares expected phonemes (Gold Standard) with actually recognized phonemes from audio and provides detailed feedback with pronunciation tips.

## Features

- **Speech-to-Text (ASR)**: Automatic text recognition from audio using OpenAI Whisper or macOS Speech framework
- **Voice Activity Detection (VAD)**: Automatically trims noise from audio with ultra-conservative settings
- **Audio Normalization**: Adaptive normalization to handle AGC (Automatic Gain Control) issues
- **G2P (Grapheme-to-Phoneme)**: Converts German text to IPA phonemes using eSpeak NG with dictionary lookup support (IPA-Dict-DSL and MFA dictionaries)
- **Phoneme Recognition**: Recognizes phonemes from audio using Wav2Vec2 XLSR-53 eSpeak CV fine-tuned model
- **Multi-level Filtering**: 
  - Whitelist filtering (German IPA phonemes + common errors)
  - Confidence score filtering (removes low-confidence predictions)
- **Forced Alignment**: Extracts precise phoneme segments with timestamps using torchaudio CTC alignment
- **MFA Alignment** (Optional): Montreal Forced Aligner integration for improved alignment accuracy
- **Needleman-Wunsch Alignment**: Matches expected and recognized phonemes using phonetic similarity matrix, handling gaps intelligently
- **Phoneme Similarity Matrix**: Uses phonetic features (voicing, place, manner, vowel properties) for intelligent alignment
- **Diagnostic Engine**: Generates feedback for pronunciation errors based on a diagnostic matrix
- **Phoneme Validation** (Optional): Two-stage validation through trained models for improved accuracy
- **Metrics**: Word Error Rate (WER) and Phoneme Error Rate (PER) calculation
- **Chat Interface**: Interactive chat-based UI with conversation history
- **Visualization**: 
  - Side-by-side comparison of expected vs recognized phonemes
  - Colored text (green = correct, red = incorrect, gray = missing)
  - Detailed report with feedback
  - Multi-model comparison views

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install eSpeak NG (for G2P):
- macOS: `brew install espeak-ng`
- Linux: `sudo apt-get install espeak-ng`
- Windows: Download from [eSpeak NG website](https://github.com/espeak-ng/espeak-ng)

3. (Optional) Install silero-vad for better VAD:
```bash
pip install silero-vad
```

4. (Optional) Install MFA (Montreal Forced Aligner) for improved alignment:
```bash
# Using conda (recommended)
conda create -n MFA310 python=3.10
conda activate MFA310
conda install -c conda-forge montreal-forced-alignment
mfa model download dictionary german_mfa
mfa model download acoustic german_mfa
```
Note: Update `MFA_CONDA_ENV` in `config.py` to match your conda environment name.

5. (Optional) Install german-phoneme-validator for two-stage validation:
```bash
# From local directory
pip install -e /path/to/german-phoneme-validator
# Or from GitHub
pip install git+https://github.com/SergejKurtasch/german-phoneme-validator.git
```

## Usage

Run the application:
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

### How to use:

1. **Enter German text** (optional): Type a German sentence in the text box, or leave it empty to use ASR for automatic transcription
2. **Record or upload audio**: Use the microphone or upload an audio file (WAV, 16kHz recommended)
3. **Enable validation** (optional): Check "Enable 2 step validation" for improved accuracy using trained models
4. **Click "Validate Pronunciation"** to see results in the chat interface

The application will:
- Recognize text from audio (if text not provided)
- Calculate Word Error Rate (WER) - if WER is too high (>70%), detailed phoneme analysis is skipped
- Extract expected phonemes from text using G2P
- Recognize phonemes from audio using Wav2Vec2
- Align expected and recognized phonemes
- Generate detailed feedback with pronunciation tips

## Configuration

Edit `config.py` to customize:

- **Model Configuration**: 
  - `MODEL_NAME`: Phoneme recognition model (default: `facebook/wav2vec2-xlsr-53-espeak-cv-ft`)
  - `MODEL_DEVICE`: Device for model inference (`auto`, `cpu`, `cuda`, `mps`)

- **ASR Settings**:
  - `ASR_ENABLED`: Enable/disable ASR functionality
  - `ASR_ENGINE`: `whisper` or `macos`
  - `ASR_MODEL`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
  - `ASR_ALWAYS_RUN`: Run ASR even when text is provided (default: False)

- **Metrics**:
  - `WER_THRESHOLD`: Skip phoneme analysis if WER exceeds this value (default: 0.70)
  - `SHOW_WER`, `SHOW_PER`: Toggle metric display

- **Phoneme Filtering**:
  - `CONFIDENCE_THRESHOLD`: Confidence threshold for filtering (default: 0.25)
  - `CONFIDENCE_THRESHOLD_UNCLEAR`: Below this, mark as "unclear" (default: 0.1)

- **MFA Settings**:
  - `MFA_ENABLED`: Enable/disable MFA alignment
  - `MFA_CONDA_ENV`: Conda environment name where MFA is installed

- **VAD Settings**: Ultra-conservative settings to avoid cutting off speech
- **Audio Normalization**: Adaptive normalization for AGC issues
- **Alignment Parameters**: Needleman-Wunsch scoring and phoneme similarity weights

## Diagnostic Matrix

The `diagnostic_matrix.json` file contains mappings from pronunciation errors to feedback messages. You can edit this file to add more error patterns and feedback.

## Phoneme Normalization

The `phoneme_normalization_table.json` file contains mappings for normalizing phoneme representations between different systems (e.g., IPA variants, ARPAbet, eSpeak).

## Project Structure

```
SpeechRec-German-diagnostic/
├── app.py                          # Main Gradio application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── diagnostic_matrix.json          # Error feedback database
├── phoneme_normalization_table.json # Phoneme normalization mappings
├── modules/
│   ├── audio_normalizer.py         # Audio normalization (AGC handling)
│   ├── vad_module.py               # Voice Activity Detection
│   ├── g2p_module.py               # G2P conversion with dictionary lookup
│   ├── speech_to_text.py           # ASR using Whisper
│   ├── speech_to_text_macos.py     # ASR using macOS Speech framework
│   ├── phoneme_recognition.py      # Phoneme recognition (Wav2Vec2)
│   ├── phoneme_filtering.py        # Multi-level filtering
│   ├── phoneme_normalizer.py       # Phoneme normalization
│   ├── phoneme_similarity.py       # Phonetic similarity matrix
│   ├── forced_alignment.py         # CTC forced alignment
│   ├── mfa_alignment.py            # MFA alignment integration
│   ├── alignment.py                # Needleman-Wunsch alignment
│   ├── diagnostic_engine.py        # Diagnostic feedback generation
│   ├── phoneme_validator.py        # Optional two-stage validation
│   ├── phoneme_validation.py       # Single phoneme validation
│   ├── metrics.py                  # WER and PER calculation
│   ├── utils.py                    # Utility functions
│   ├── component_manager.py        # Component initialization and management
│   ├── chat_utils.py               # Chat history management
│   ├── ui/
│   │   ├── styles.py               # Gradio CSS styles
│   │   └── __init__.py
│   └── visualization/
│       ├── __init__.py
│       ├── html_generators.py      # HTML generation for visualizations
│       ├── report_generators.py    # Report generation
│       ├── multi_model_comparison.py # Multi-model comparison views
│       └── helpers.py              # Visualization helpers
├── data/
│   ├── audio/                      # Sample audio files
│   └── dictionaries/               # IPA and MFA dictionaries
└── notebooks/                      # Analysis and testing notebooks
```

## Dependencies

- Python 3.10+
- **Core ML**: torch (>=2.0.0), torchaudio (>=2.0.0), transformers (>=4.30.0)
- **Audio Processing**: librosa (>=0.10.0), soundfile (>=0.12.0), numpy (>=1.24.0), scipy (>=1.10.0)
- **G2P**: phonemizer (>=3.0.0)
- **Alignment**: biopython (>=1.81), textgrid (>=1.5.0)
- **VAD**: silero-vad (>=4.0.0)
- **ASR**: openai-whisper (>=20231117), pyobjc-framework-Speech (>=9.0, macOS only)
- **Metrics**: jiwer (>=3.0.0)
- **Frontend**: gradio (>=4.0.0)
- **Optional**: MFA (via conda), german-phoneme-validator

## Notes

- **Model Configuration**: The application uses `facebook/wav2vec2-xlsr-53-espeak-cv-ft` by default. This model is fine-tuned on Common Voice for phoneme recognition and outputs phonetic labels directly via Wav2Vec2ForCTC.

- **ASR Integration**: When text is not provided, the application automatically transcribes audio using Whisper (or macOS Speech on macOS). ASR can be disabled or configured in `config.py`.

- **WER Threshold**: If the Word Error Rate exceeds 70% (configurable), detailed phoneme analysis is skipped and only text comparison is shown. This improves performance when the recognized text differs significantly from the expected text.

- **MFA Alignment**: MFA (Montreal Forced Aligner) provides more accurate alignment than CTC-only alignment but requires installation and configuration. It's optional and can be enabled in `config.py`.

- **Audio Normalization**: The application includes adaptive audio normalization to handle AGC (Automatic Gain Control) issues that can affect phoneme recognition accuracy.

- **Phoneme Similarity**: The alignment algorithm uses a phonetic feature-based similarity matrix, considering voicing, place of articulation, manner of articulation, and vowel properties for intelligent matching.

- **Dictionary Support**: The application supports multiple dictionary sources:
  - IPA-Dict-DSL (primary): Better for loanwords
  - MFA Dictionary (fallback): Standard German phoneme dictionary

- All feedback is in English by default (can be extended to other languages)

## Testing

To test the application:

1. **Basic functionality test:**
   - Start the application: `python app.py`
   - Enter a simple German sentence: "Hallo, wie geht es dir?"
   - Record or upload audio
   - Click "Validate Pronunciation"
   - Check that all outputs are displayed correctly in the chat

2. **ASR test:**
   - Leave text input empty
   - Record or upload audio
   - Verify that text is automatically transcribed
   - Check that phoneme analysis uses transcribed text

3. **Test cases:**
   - **Correct pronunciation**: All phonemes should match (green in colored text)
   - **Typical errors**: Test with common errors (e.g., saying "u" instead of "ʏ")
   - **Missing phonemes**: Test with incomplete pronunciation
   - **Noise in audio**: Test with noisy audio to verify filtering and normalization
   - **High WER**: Test with very different text to verify WER threshold behavior

4. **Component testing:**
   - VAD: Verify that silence is trimmed (ultra-conservative settings)
   - Audio normalization: Check that AGC issues are handled
   - G2P: Check that expected phonemes are extracted correctly
   - ASR: Verify automatic transcription
   - Phoneme recognition: Verify that phonemes are recognized from audio
   - Filtering: Check that low-confidence and non-whitelist phonemes are filtered
   - Alignment: Verify that expected and recognized phonemes are aligned correctly
   - Diagnostic: Check that feedback is generated for errors

## Troubleshooting

- **VAD not working**: Install silero-vad: `pip install silero-vad`
- **G2P not working**: Install eSpeak NG (see Installation section)
- **ASR not working**: 
  - For Whisper: Check internet connection (downloads model on first use)
  - For macOS Speech: Ensure microphone permissions are granted
  - Check `ASR_ENGINE` and `ASR_MODEL` settings in `config.py`
- **Forced alignment errors**: Check torchaudio version (should be >= 2.0.0)
- **MFA alignment errors**:
  - Ensure MFA is installed in the correct conda environment
  - Check `MFA_CONDA_ENV` matches your environment name
  - Verify dictionaries and acoustic models are downloaded: `mfa model list`
- **Model loading errors**: 
  - Check internet connection (model downloads from Hugging Face)
  - If you see "model is not a valid model identifier":
    - The model name may be incorrect or the model may not exist
    - Check the model name on https://huggingface.co/models
    - If the model is private, authenticate with: `huggingface-cli login`
    - Update `MODEL_NAME` in `config.py` to use a valid model
- **Component initialization errors**: Check that all required dependencies are installed
- **Dictionary loading errors**: Dictionaries are downloaded automatically. If issues occur, check `DICTIONARY_DIR` path in `config.py`

## License

See main project license.

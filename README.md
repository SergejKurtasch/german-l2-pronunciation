# German Pronunciation Diagnostic App (L2-Trainer)

A diagnostic application for learning German pronunciation that compares expected phonemes (Gold Standard) with actually recognized phonemes from audio and provides detailed feedback with pronunciation tips.

## Features

- **Voice Activity Detection (VAD)**: Automatically trims noise from audio
- **G2P (Grapheme-to-Phoneme)**: Converts German text to IPA phonemes using eSpeak NG
- **Phoneme Recognition**: Recognizes phonemes from audio using Wav2Vec2 XLS-R 21 Phonemes model
- **Multi-level Filtering**: 
  - Whitelist filtering (German IPA phonemes + common errors)
  - Confidence score filtering (removes low-confidence predictions)
- **Forced Alignment**: Extracts precise phoneme segments with timestamps using torchaudio
- **Needleman-Wunsch Alignment**: Matches expected and recognized phonemes, handling gaps
- **Diagnostic Engine**: Generates feedback for pronunciation errors based on a diagnostic matrix
- **Optional Validation**: Additional validation through trained models (if available from main project)
- **Visualization**: 
  - Side-by-side comparison of expected vs recognized phonemes
  - Colored text (green = correct, red = incorrect, gray = missing)
  - Detailed report with feedback

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

## Usage

Run the application:
```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

### How to use:

1. Enter a German sentence in the text box
2. Record audio using the microphone or upload an audio file (WAV, 16kHz recommended)
3. Optionally enable additional validation through trained models
4. Click "Validate Pronunciation" to see results

## Configuration

Edit `config.py` to customize:
- Model name and device
- Confidence thresholds
- VAD settings
- Alignment parameters
- Phoneme whitelist

## Diagnostic Matrix

The `diagnostic_matrix.json` file contains mappings from pronunciation errors to feedback messages. You can edit this file to add more error patterns and feedback.

## Project Structure

```
SpeechRec-German-diagnostic/
├── app.py                          # Main application
├── requirements.txt                # Dependencies
├── diagnostic_matrix.json          # Error feedback database
├── config.py                       # Configuration
├── modules/
│   ├── vad_module.py              # Voice Activity Detection
│   ├── g2p_module.py              # G2P conversion
│   ├── phoneme_recognition.py     # Phoneme recognition
│   ├── phoneme_filtering.py       # Multi-level filtering
│   ├── forced_alignment.py        # Forced alignment
│   ├── alignment.py               # Needleman-Wunsch alignment
│   ├── diagnostic_engine.py       # Diagnostic feedback
│   ├── phoneme_validator.py       # Optional validation
│   └── visualization.py           # Visualization
└── README.md
```

## Dependencies

- Python 3.10+
- torch, torchaudio
- transformers
- librosa, soundfile
- phonemizer
- biopython (optional, for Needleman-Wunsch)
- silero-vad or webrtcvad (for VAD)
- gradio

## Notes

- **Model Configuration**: The application is configured to use `facebook/wav2vec2-xls-r-300m` by default. 
  - **Important**: The model `facebook/wav2vec2-xls-r-300m-21-phonemes` does not exist on Hugging Face.
  - For phoneme recognition, you may need to:
    1. Use a fine-tuned model trained on phonemes
    2. Fine-tune the base model yourself on a phoneme dataset
    3. Use a custom model with a phoneme tokenizer
  - You can change the model in `config.py` by setting `MODEL_NAME` to your preferred model.
- Forced alignment requires torchaudio's `forced_align` function
- Optional validation can use trained models from the main SpeechRec-German project if available
- All feedback is in English by default (can be extended to other languages)

## Testing

To test the application:

1. **Basic functionality test:**
   - Start the application: `python app.py`
   - Enter a simple German sentence: "Hallo, wie geht es dir?"
   - Record or upload audio
   - Click "Validate Pronunciation"
   - Check that all outputs are displayed correctly

2. **Test cases:**
   - **Correct pronunciation**: All phonemes should match (green in colored text)
   - **Typical errors**: Test with common errors (e.g., saying "u" instead of "ʏ")
   - **Missing phonemes**: Test with incomplete pronunciation
   - **Noise in audio**: Test with noisy audio to verify filtering

3. **Component testing:**
   - VAD: Verify that silence is trimmed
   - G2P: Check that expected phonemes are extracted correctly
   - Phoneme recognition: Verify that phonemes are recognized from audio
   - Filtering: Check that low-confidence and non-whitelist phonemes are filtered
   - Alignment: Verify that expected and recognized phonemes are aligned correctly
   - Diagnostic: Check that feedback is generated for errors

## Troubleshooting

- **VAD not working**: Install silero-vad: `pip install silero-vad`
- **G2P not working**: Install eSpeak NG (see Installation section)
- **Forced alignment errors**: Check torchaudio version (should be >= 2.0.0)
- **Model loading errors**: 
  - Check internet connection (model downloads from Hugging Face)
  - If you see "model is not a valid model identifier":
    - The model name may be incorrect or the model may not exist
    - Check the model name on https://huggingface.co/models
    - If the model is private, authenticate with: `huggingface-cli login`
    - Update `MODEL_NAME` in `config.py` to use a valid model
  - For phoneme recognition, ensure the model is trained for phoneme recognition or has a phoneme tokenizer

## License

See main project license.


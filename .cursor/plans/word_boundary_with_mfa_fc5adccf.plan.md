---
name: Word Boundary with MFA
overview: ""
todos:
  - id: copy-mfa-aligner
    content: Скопировать и адаптировать mfa_aligner.py из родительского проекта
    status: pending
  - id: create-boundary-mapper
    content: Создать функцию для маппинга MFA timestamps в word boundaries
    status: pending
  - id: create-test-notebook
    content: Создать тестовый ноутбук word_boundary_forced_alignment_test.ipynb
    status: pending
  - id: test-mfa-alignment
    content: Протестировать получение word timestamps из MFA на примерах
    status: pending
  - id: test-phoneme-mapping
    content: Протестировать сопоставление phonemes с MFA timestamps
    status: pending
  - id: test-edge-cases
    content: Проверить edge cases (одинаковые фонемы на стыках, пунктуация, заимствования)
    status: pending
  - id: update-word-boundary-utils
    content: Обновить modules/word_boundary_utils.py с новой MFA функцией
    status: pending
  - id: update-main-notebook
    content: Обновить deep_phoneme_mismatch_analysis.ipynb для использования MFA
    status: pending
  - id: create-documentation
    content: Создать документацию FORCED_ALIGNMENT_MFA_GUIDE.md
    status: pending
  - id: run-full-analysis
    content: Запустить полный анализ с ANALYSIS_STAGE=2 и проверить результаты
    status: pending
---

# План: Forced Alignment для границ слов с Montreal Forced Aligner

## Обновление: Обнаружена существующая инфраструктура ✅

**Отличная новость!** В проекте уже есть:

- ✅ MFA установлен в окружении `mfa310` (`/Volumes/SSanDisk/SpeechRec-German/miniforge/envs/mfa310/bin/mfa`)
- ✅ Готовый модуль `mfa_aligner.py` в родительском проекте `SpeechRec-German`
- ✅ Немецкий словарь MFA в `data/dictionaries/german_mfa.dict`
- ✅ Существующий `modules/forced_alignment.py` с CTC-based alignment

Это **значительно упрощает** реализацию!

---

## Проблема

Текущий алгоритм вставки `||` использует **пропорциональное распределение**, которое не учитывает ошибки распознавания. Результат - разделители попадают в неправильные места.

**Пример проблемы:**

```
Text: "Verschwenden Sie nicht länger meine Lebenszeit!"
Expected:   f ɛ ʁ ʃ v ɛ n d ə n || z iː || n ɪ ç t || ...
Recognized: f ɛ ɾ ʃ v ɛ n d ə n z iː n ɪ ç t ...  (нет ||)
```

## Решение

Использовать **Montreal Forced Aligner (MFA)** для определения точных временных меток каждого слова в аудио, затем сопоставить распознанные фонемы с этими метками.

## Архитектура решения

```mermaid
flowchart TD
    Audio[Audio WAV] --> MFA[Montreal Forced Aligner]
    Text[Text Sentence] --> MFA
    Dict[german_mfa.dict] --> MFA
    MFA --> WordTimestamps[Word Timestamps from TextGrid]
    
    Audio --> Wav2Vec2[Wav2Vec2 Model]
    Wav2Vec2 --> Logits[Logits Tensor]
    Logits --> RecPhonemes[Recognized Phonemes]
    Logits --> FrameTimes[Frame-level Timestamps]
    
    WordTimestamps --> Mapper[Phoneme-to-Word Mapper]
    RecPhonemes --> Mapper
    FrameTimes --> Mapper
    
    Mapper --> Result[Phonemes with || boundaries]
```

## Упрощённые этапы реализации

### 1. Копирование MFA модуля (УПРОЩЕНО ✅)

**Файл:** [`modules/mfa_aligner.py`](modules/mfa_aligner.py) (копия из родительского проекта)

**Действия:**

1. Скопировать `mfa_aligner.py` из `/Volumes/SSanDisk/SpeechRec-German/gradio_modules/mfa_aligner.py`
2. Адаптировать пути для текущего проекта
3. Убедиться, что путь к MFA binary корректный: `/Volumes/SSanDisk/SpeechRec-German/miniforge/envs/mfa310/bin/mfa`

**Ключевые функции из существующего модуля:**

- `MFAAligner.__init__()` - инициализация с автоопределением MFA binary
- `MFAAligner.align_single_file()` - выравнивание одного файла, возвращает phoneme timestamps
- `MFAAligner._parse_textgrid()` - парсинг TextGrid файла

### 2. Создание функции маппинга word boundaries

**Файл:** [`modules/word_boundary_utils.py`](modules/word_boundary_utils.py) (обновить существующий)

Добавить новую функцию:

```python
def insert_word_boundaries_mfa(
    text: str,
    recognized_phonemes: List[str],
    audio_path: str,
    logits: torch.Tensor,
    audio_duration: float,
    sample_rate: int = 16000,
    mfa_aligner=None
) -> List[str]:
    """
    Insert word boundaries using MFA timestamps.
    
    Algorithm:
    1. Use MFA to get word-level timestamps from audio + text
    2. Map recognized_phonemes to frame-level timestamps using logits
    3. Insert || at positions where phoneme timestamps cross word boundaries
    4. PRESERVE all phonemes (no collapsing)
    
    Args:
        text: Original text
        recognized_phonemes: List of recognized phonemes (without ||)
        audio_path: Path to audio file
        logits: Logits tensor from Wav2Vec2 (for frame-level timing)
        audio_duration: Duration of audio in seconds
        sample_rate: Audio sample rate
        mfa_aligner: MFAAligner instance (will create if None)
    
    Returns:
        List of phonemes with || inserted at word boundaries
    """
    from modules.mfa_aligner import get_mfa_aligner
    
    if mfa_aligner is None:
        mfa_aligner = get_mfa_aligner(
            mfa_bin="/Volumes/SSanDisk/SpeechRec-German/miniforge/envs/mfa310/bin/mfa",
            mfa_dict="german_mfa",  # Uses data/dictionaries/german_mfa.dict
            mfa_model="german_mfa"
        )
    
    # Step 1: Get word boundaries from MFA
    try:
        mfa_phonemes = mfa_aligner.align_single_file(audio_path, text)
        word_boundaries = _extract_word_boundaries_from_mfa(text, mfa_phonemes)
    except Exception as e:
        print(f"MFA failed: {e}, falling back to proportional distribution")
        return insert_word_boundaries(text, expected_phonemes, recognized_phonemes)
    
    # Step 2: Map recognized phonemes to timestamps
    phoneme_timestamps = _map_phonemes_to_timestamps(
        recognized_phonemes, logits, audio_duration, sample_rate
    )
    
    # Step 3: Insert || based on word boundaries
    result = _insert_boundaries_at_timestamps(
        phoneme_timestamps, word_boundaries
    )
    
    return result


def _extract_word_boundaries_from_mfa(
    text: str,
    mfa_phonemes: List[Dict]
) -> List[float]:
    """
    Extract word boundary timestamps from MFA alignment.
    
    Returns:
        List of timestamps (in seconds) where word boundaries occur
    """
    # MFA returns phoneme-level timestamps with start_ms/end_ms
    # We need to identify word boundaries
    
    words = text.split()
    boundaries = []
    
    # Group MFA phonemes by words based on cumulative phoneme count
    # (MFA aligns phonemes to text, so we can map them to words)
    
    # ... implementation ...
    return boundaries


def _map_phonemes_to_timestamps(
    recognized_phonemes: List[str],
    logits: torch.Tensor,
    audio_duration: float,
    sample_rate: int
) -> List[Tuple[str, float]]:
    """
    Map each recognized phoneme to its approximate timestamp.
    
    Uses CTC decoding path to find frame indices for each phoneme.
    """
    # Wav2Vec2 outputs logits at ~50 fps (320 samples stride at 16kHz)
    num_frames = logits.shape[1]
    frame_duration = audio_duration / num_frames
    
    # Use greedy CTC decoding to find phoneme boundaries
    # ... implementation similar to forced_alignment.py ...
    
    return phoneme_timestamps


def _insert_boundaries_at_timestamps(
    phoneme_timestamps: List[Tuple[str, float]],
    word_boundaries: List[float]
) -> List[str]:
    """
    Insert || at word boundary timestamps.
    
    IMPORTANT: Preserve all phonemes, even if identical at boundaries.
    """
    result = []
    boundary_idx = 0
    
    for i, (phoneme, timestamp) in enumerate(phoneme_timestamps):
        # Check if we've crossed a word boundary
        if boundary_idx < len(word_boundaries):
            if timestamp >= word_boundaries[boundary_idx]:
                # Insert boundary BEFORE current phoneme
                if result and result[-1] != '||':
                    result.append('||')
                boundary_idx += 1
        
        result.append(phoneme)
    
    return result
```

### 3. Создание тестового ноутбука

**Файл:** [`notebooks/word_boundary_forced_alignment_test.ipynb`](notebooks/word_boundary_forced_alignment_test.ipynb) (новый)

**Структура:**

#### Cell 1: Setup & Imports

```python
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

from modules.mfa_aligner import get_mfa_aligner
from modules.word_boundary_utils import insert_word_boundaries_mfa
from modules.phoneme_recognition import PhonemeRecognizer
from modules.g2p_module import get_g2p
import pandas as pd
```

#### Cell 2: Initialize MFA

```python
# Initialize MFA aligner with explicit path
mfa_aligner = get_mfa_aligner(
    mfa_bin="/Volumes/SSanDisk/SpeechRec-German/miniforge/envs/mfa310/bin/mfa",
    mfa_dict="german_mfa",
    mfa_model="german_mfa"
)

print("MFA aligner initialized!")
print(f"MFA binary: {mfa_aligner.mfa_bin}")
```

#### Cell 3: Load test data

```python
# Load a few examples from metadata
df = pd.read_csv('/Volumes/SSanDisk/SpeechRec-German-diagnostic/data/dictionaries/metadata_wav_clean_hochdeutsch.csv')

# Fix audio paths
df['audio_path_fixed'] = df['audio_wav_path'].str.replace(
    '/Volumes/SSanDisk/SpeechRec-German/',
    '/Volumes/SSanDisk/audio_data/'
)

# Sample 10 records for testing
test_df = df.sample(n=10, random_state=42)
test_df
```

#### Cell 4: Test MFA alignment on single file

```python
# Test on first record
test_record = test_df.iloc[0]

print(f"Text: {test_record['text']}")
print(f"Audio: {test_record['audio_path_fixed']}")

# Get MFA alignment
mfa_result = mfa_aligner.align_single_file(
    test_record['audio_path_fixed'],
    test_record['text']
)

print(f"\nMFA phonemes ({len(mfa_result)}):")
for p in mfa_result[:20]:
    print(f"  {p['phoneme']:5s} {p['start_ms']:7.1f} - {p['end_ms']:7.1f} ms")
```

#### Cell 5: Test full pipeline

```python
recognizer = PhonemeRecognizer()
g2p = get_g2p()

for i, row in test_df.iterrows():
    text = row['text']
    audio_path = row['audio_path_fixed']
    
    print(f"\n{'='*80}")
    print(f"Text: {text}")
    
    # Get expected phonemes
    expected_dict = g2p.process_sentence(text)
    expected_phonemes = [p.get('phoneme', '') for p in expected_dict if p.get('phoneme')]
    
    # Get recognized phonemes
    logits, _ = recognizer.recognize_phonemes(audio_path)
    recognized_str = recognizer.decode_phonemes(logits)
    recognized_phonemes = recognized_str.split()
    
    # OLD method (proportional)
    from modules.word_boundary_utils import insert_word_boundaries as old_method
    old_result = old_method(text, expected_phonemes, recognized_phonemes)
    
    # NEW method (MFA)
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    audio_duration = waveform.shape[1] / sr
    
    new_result = insert_word_boundaries_mfa(
        text=text,
        recognized_phonemes=recognized_phonemes,
        audio_path=audio_path,
        logits=logits,
        audio_duration=audio_duration,
        mfa_aligner=mfa_aligner
    )
    
    # Compare
    print(f"\nExpected:   {' '.join(expected_phonemes)}")
    print(f"OLD method: {' '.join(old_result)}")
    print(f"NEW method: {' '.join(new_result)}")
    
    # Count || positions
    expected_boundaries = [i for i, p in enumerate(expected_phonemes) if p == '||']
    old_boundaries = [i for i, p in enumerate(old_result) if p == '||']
    new_boundaries = [i for i, p in enumerate(new_result) if p == '||']
    
    print(f"\nBoundaries:")
    print(f"  Expected: {len(expected_boundaries)} at positions {expected_boundaries}")
    print(f"  OLD:      {len(old_boundaries)} at positions {old_boundaries}")
    print(f"  NEW:      {len(new_boundaries)} at positions {new_boundaries}")
```

#### Cell 6: Edge cases testing

```python
# Test edge cases
edge_cases = [
    {
        'text': 'an Nadine',  # Two 'n' at boundary
        'audio': '...'
    },
    {
        'text': 'Das ist ein Computer',  # Loanword
        'audio': '...'
    },
    {
        'text': 'Hallo, wie geht\'s?',  # Punctuation
        'audio': '...'
    }
]

# Test each case...
```

#### Cell 7: Evaluation metrics

```python
# Calculate boundary accuracy
def calculate_boundary_accuracy(expected, predicted):
    expected_positions = set(i for i, p in enumerate(expected) if p == '||')
    predicted_positions = set(i for i, p in enumerate(predicted) if p == '||')
    
    if not expected_positions:
        return 1.0 if not predicted_positions else 0.0
    
    # Allow ±1 position tolerance
    correct = 0
    for exp_pos in expected_positions:
        if any(abs(exp_pos - pred_pos) <= 1 for pred_pos in predicted_positions):
            correct += 1
    
    return correct / len(expected_positions)

# Calculate for OLD vs NEW
old_accuracies = []
new_accuracies = []

# ... (run on test set and compare) ...

print(f"Average boundary accuracy:")
print(f"  OLD method: {np.mean(old_accuracies):.2%}")
print(f"  NEW method: {np.mean(new_accuracies):.2%}")
```

### 4. Интеграция в основной пайплайн

После успешного тестирования:

#### 4.1 Обновить [`notebooks/deep_phoneme_mismatch_analysis.ipynb`](notebooks/deep_phoneme_mismatch_analysis.ipynb)

Добавить конфигурационную ячейку (после Cell 1):

```python
# =============================================================================
# CONFIGURATION
# =============================================================================
USE_MFA_ALIGNMENT = True  # Toggle NEW (MFA) vs OLD (proportional) algorithm
ANALYSIS_STAGE = 2  # Stage 2: MFA-based word boundaries
RANDOM_SEED = ANALYSIS_STAGE * 100
STAGE_SUFFIX = f"_stage{ANALYSIS_STAGE}"

# MFA configuration
MFA_BIN = "/Volumes/SSanDisk/SpeechRec-German/miniforge/envs/mfa310/bin/mfa"
MFA_DICT = "german_mfa"
MFA_MODEL = "german_mfa"
```

Обновить функцию `extract_phonemes_for_record`:

```python
def extract_phonemes_for_record(row):
    # ... (existing code) ...
    
    # STEP 1: Insert word boundaries
    if USE_MFA_ALIGNMENT:
        # NEW: Use MFA for accurate word boundaries
        recognized_phonemes_with_boundaries = insert_word_boundaries_mfa(
            text=text,
            recognized_phonemes=recognized_phonemes_raw,
            audio_path=audio_path,
            logits=logits,
            audio_duration=audio_duration,
            mfa_aligner=mfa_aligner  # Initialize once at notebook level
        )
    else:
        # OLD: Use proportional distribution
        recognized_phonemes_with_boundaries = insert_word_boundaries(
            text, 
            expected_phonemes_raw, 
            recognized_phonemes_raw
        )
    
    # ... (rest of code) ...
```

Инициализировать MFA aligner один раз (в Cell после импортов):

```python
if USE_MFA_ALIGNMENT:
    from modules.mfa_aligner import get_mfa_aligner
    mfa_aligner = get_mfa_aligner(
        mfa_bin=MFA_BIN,
        mfa_dict=MFA_DICT,
        mfa_model=MFA_MODEL
    )
    print("✓ MFA aligner initialized")
else:
    mfa_aligner = None
```

### 5. Зависимости

**Файл:** [`requirements.txt`](requirements.txt)

Добавить (если ещё нет):

```txt
# Montreal Forced Aligner support
praatio>=5.0.0  # For parsing TextGrid files
```

**Примечание:** MFA уже установлен в окружении `mfa310`, не нужно добавлять в requirements.

### 6. Документация

**Файл:** [`.cursor/reports/FORCED_ALIGNMENT_MFA_GUIDE.md`](.cursor/reports/FORCED_ALIGNMENT_MFA_GUIDE.md) (новый)

Содержание:

- Обзор решения (MFA для word boundaries)
- Как использовать (включить `USE_MFA_ALIGNMENT = True`)
- Архитектура (диаграмма потока данных)
- Сравнение OLD vs NEW подхода
- Troubleshooting (MFA timeout, missing TextGrid, etc.)
- Пример результатов

## Ключевые преимущества упрощённого подхода

1. **✅ Не нужна установка MFA** - уже установлен в `mfa310`
2. **✅ Готовый код** - `mfa_aligner.py` уже написан и протестирован
3. **✅ Словарь готов** - `german_mfa.dict` уже в проекте
4. **✅ Быстрая интеграция** - копирование модуля + создание маппинга
5. **✅ Fallback механизм** - если MFA не работает, используется старый алгоритм

## Тестирование

1. ✅ Проверить, что MFA binary доступен и работает
2. Запустить `word_boundary_forced_alignment_test.ipynb`
3. Проверить визуально 10-20 примеров
4. Убедиться, что `||` вставляются корректно
5. Проверить, что фонемы на стыках НЕ теряются
6. Сравнить точность OLD vs NEW метода
7. Если всё ОК - обновить основной ноутбук с `ANALYSIS_STAGE = 2`
8. Запустить полный анализ 1000 записей

## Критерий успеха

**До (OLD - пропорциональное распределение):**

```
Expected:   f ɛ ʁ ʃ v ɛ n d ə n || z iː || n ɪ ç t || l ɛ ŋ ɐ || m aɪ n ə || l e b ɛ n s t s ə iː t
Recognized: f ɛ ɾ ʃ v ɛ n d ə || n z || iː n ɪ ç || t l ɛ ŋ || ...  (неправильно)
```

**После (NEW - MFA-based):**

```
Expected:   f ɛ ʁ ʃ v ɛ n d ə n || z iː || n ɪ ç t || l ɛ ŋ ɐ || m aɪ n ə || l e b ɛ n s t s ə iː t
Recognized: f ɛ ɾ ʃ v ɛ n d ə n || z iː || n ɪ ç t || l ɛ ŋ ɜ || m aɪ n ə || l eː b ə n s ts aɪ t
                        ✓ граница!    ✓ граница!   ✓ граница!      ✓ граница!
```

**Ожидаемый результат:**

- Разделители `||` совпадают по словам (±1 фонема допустимо)
- Все фонемы сохранены (нет потерь на границах)
- Accuracy word boundaries: >85% (vs ~50% у старого метода)
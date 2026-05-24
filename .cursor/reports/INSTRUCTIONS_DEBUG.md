# Инструкции по отладке ошибки обработки фонем

## Проблема
Все 1000 записей завершились с ошибками при извлечении фонем из аудио.

## Шаги для диагностики

### 1. Запустите отладочную ячейку в notebook

Откройте `deep_phoneme_mismatch_analysis.ipynb` и найдите новую ячейку **"3.1 Debug: Test Single File Processing"**.

Запустите эту ячейку - она попробует обработать один файл и покажет детальную ошибку.

### 2. Проверьте, что показывает ошибка

Наиболее вероятные причины:

#### A. Проблема с устройством (GPU/CPU)
```
RuntimeError: CUDA out of memory
```
**Решение**: Модель пытается использовать GPU, но памяти недостаточно.

Добавьте в начало notebook (после импортов):
```python
import torch
# Force CPU usage
recognizer = PhonemeRecognizer(device='cpu')
```

#### B. Проблема с моделью
```
OSError: Can't load model
```
**Решение**: Модель не загружена или кэш поврежден.

Проверьте интернет-соединение и перезагрузите модель:
```python
recognizer = PhonemeRecognizer(model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
```

#### C. Проблема с librosa
```
ModuleNotFoundError: No module named 'librosa'
```
**Решение**: Установите недостающие зависимости:
```bash
pip install librosa soundfile
```

#### D. Проблема с форматом аудио
```
LibsndfileError: Error opening
```
**Решение**: Файл поврежден или неправильный формат.

Проверьте файлы:
```python
import librosa
test_audio, sr = librosa.load(df_sample.iloc[0]['audio_path_fixed'], sr=16000)
print(f"Loaded: {len(test_audio)} samples at {sr} Hz")
```

### 3. Отправьте полную ошибку

Если ни одно из решений не помогло, скопируйте **полный traceback** из отладочной ячейки и отправьте мне.

## Временное решение

Если хотите продолжить анализ на меньшем количестве файлов, измените:

```python
# Вместо 1000 файлов, попробуйте 10
n_regular = min(10, len(df_regular))  # было 900
n_loanword = min(2, len(df_loanwords))  # было 100
```

Это позволит быстрее найти проблему.

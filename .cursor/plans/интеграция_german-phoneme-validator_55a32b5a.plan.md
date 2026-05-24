---
name: Интеграция german-phoneme-validator
overview: Интеграция модуля german-phoneme-validator для вторичной проверки Fonem с обновлением визуализации (зеленый цвет) при confidence > 70%
todos:
  - id: update-validator-import
    content: Обновить импорт в modules/phoneme_validator.py для использования german-phoneme-validator
    status: completed
  - id: update-validation-logic
    content: "Улучшить Stage 9 в app.py: добавить проверку confidence > 70% и обновление aligned_pairs"
    status: completed
  - id: fix-phoneme-pair-detection
    content: Исправить получение phoneme_pair через validator.get_phoneme_pair()
    status: completed
  - id: improve-segment-matching
    content: Улучшить сопоставление audio segments с aligned_pairs для корректной валидации
    status: completed
---

# Интеграция german-phoneme-validator в проект

## Цель

Добавить второй этап проверки Fonem через обученные модели из проекта `german-phoneme-validator`. Если валидация показывает, что Fonem была правильной с вероятностью > 70%, отметить ее зеленым цветом во всех визуализациях.

## Текущее состояние

1. **Checkbox уже существует** в `app.py` (строка 1112-1116) с параметром `enable_validation`
2. **Модуль `phoneme_validator.py`** пытается импортировать из неправильного пути (`SpeechRec-German`)
3. **Stage 9: Optional Validation** (строки 792-834) уже есть, но:

                        - Не проверяет confidence > 70%
                        - Не обновляет `aligned_pairs` (только `diagnostic_results`)
                        - Использует неправильный путь для получения phoneme_pair

## Изменения

### 1. Обновить `modules/phoneme_validator.py`

**Файл:** `modules/phoneme_validator.py`

- Заменить импорт из `SpeechRec-German` на импорт из `german-phoneme-validator`
- Путь к проекту: `/Volumes/SSanDisk/german-phoneme-validator`
- Использовать `get_validator()` из `german-phoneme-validator`
- Обновить методы `has_trained_model()` и `validate_phoneme_segment()` для работы с новым API

**Ключевые изменения:**

```python
# Старый путь (неправильный):
parent_project = Path(__file__).parent.parent.parent / "SpeechRec-German"

# Новый путь:
validator_project = Path(__file__).parent.parent.parent / "german-phoneme-validator"
sys.path.insert(0, str(validator_project))
from core.validator import PhonemeValidator, get_validator
```

### 2. Улучшить Stage 9: Optional Validation в `app.py`

**Файл:** `app.py` (строки 792-834)

**Изменения:**

1. **Проверка confidence > 70%:**

                        - Добавить проверку `validation_result.get('confidence', 0.0) > 0.7`
                        - Обновлять результат только если confidence достаточна

2. **Обновление `aligned_pairs`:**

                        - После успешной валидации обновить соответствующий элемент в `aligned_pairs`
                        - Если валидация показывает, что `recognized_ph` на самом деле правильный, изменить пару на `(expected_ph, expected_ph)` для визуализации

3. **Правильное получение phoneme_pair:**

                        - Использовать `optional_validator.validator.get_phoneme_pair(expected_ph, recognized_ph)` вместо ручного создания строки

4. **Обработка всех несовпадающих пар:**

                        - Итерировать по `aligned_pairs` напрямую, а не только по `diagnostic_results`
                        - Для каждой несовпадающей пары проверять наличие модели

**Логика обновления:**

```python
# После успешной валидации с confidence > 70%:
if validation_result.get('is_correct') and validation_result.get('confidence', 0.0) > 0.7:
    # Обновить diagnostic_results
    result['is_correct'] = True
    result['validation_confidence'] = validation_result.get('confidence', 0.0)
    
    # Обновить aligned_pairs для визуализации
    # Найти индекс в aligned_pairs и изменить recognized на expected
    for idx, (exp, rec) in enumerate(aligned_pairs):
        if exp == expected_ph and rec == recognized_ph:
            # Изменить recognized на expected для зеленого цвета
            aligned_pairs[idx] = (exp, exp)
            break
```

### 3. Обработка сегментов аудио

**Проблема:** Текущий код ищет сегмент по `seg.label == recognized_ph`, но это может не работать, если сегментов несколько с одинаковым label.

**Решение:**

- Использовать индекс из `aligned_pairs` для сопоставления с `recognized_segments`
- Или использовать временные метки из forced alignment

### 4. Обновление визуализации

Визуализация автоматически обновится, так как:

- `create_side_by_side_comparison()` использует `aligned_pairs` - если мы обновим пару на `(exp, exp)`, она будет зеленой
- `create_colored_text()` использует `aligned_pairs_tuples` - также обновится автоматически
- `diagnostic_results` используется для детального отчета - также обновится

## Структура изменений

```
modules/phoneme_validator.py
 - Обновить импорт (german-phoneme-validator вместо SpeechRec-German)
 - Обновить методы для работы с новым API

app.py
 - Stage 9: Optional Validation (строки 792-834)
  - Добавить проверку confidence > 70%
  - Обновлять aligned_pairs после успешной валидации
  - Использовать правильный метод для получения phoneme_pair
  - Улучшить сопоставление сегментов с aligned_pairs
```

## Тестирование

После изменений проверить:

1. Checkbox работает и включает валидацию
2. Для несовпадающих пар с доступными моделями вызывается валидация
3. При confidence > 70% Fonem отмечается зеленым в:

                        - Side-by-side comparison
                        - Colored text
                        - Detailed report

4. Визуализация обновляется корректно для всех компонентов

## Зависимости

- Проект `german-phoneme-validator` должен быть доступен по пути `/Volumes/SSanDisk/german-phoneme-validator`
- Модели должны быть в `german-phoneme-validator/artifacts/`
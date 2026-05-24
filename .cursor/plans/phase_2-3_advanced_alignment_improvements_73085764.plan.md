---
name: Phase 2-3 Advanced Alignment Improvements
overview: "Средний и долгосрочный приоритет: обработка составных слов, постобработка, заимствования и визуализация"
todos:
  - id: phase2_create_dicts
    content: "Создать словари: proper_nouns.dict и loanwords.dict"
    status: pending
  - id: phase2_word_type
    content: Реализовать функцию get_word_type() в modules/alignment.py
    status: pending
  - id: phase2_modify_align
    content: Модифицировать needleman_wunsch_align() для поддержки типов слов
    status: pending
  - id: phase2_postprocessing
    content: Создать модуль alignment_postprocessing.py с функциями постобработки
    status: pending
  - id: phase2_integrate
    content: Интегрировать постобработку в app.py
    status: pending
  - id: phase3_loanword_rules
    content: Добавить специальные правила схожести для заимствований
    status: pending
  - id: phase3_visualization
    content: Улучшить визуализацию в modules/visualization.py
    status: pending
  - id: phase3_debug
    content: Добавить отладочный режим в config.py и логирование
    status: pending
---

# Фазы 2-3: Продвинутые улучшения выравнивания

## Обзор

Этот план включает улучшения среднего и долгосрочного приоритета, которые требуют более глубоких изменений в архитектуре системы.

## Фаза 2: Средний приоритет (1-2 недели)

### Приоритет 1: Обработка составных слов и имён

#### 1.1 Определение составных слов

**Файлы для изменения:**

- `modules/g2p_module.py` - добавление логики определения составных слов
- `modules/alignment.py` - добавление функции определения типа слова

**Подход:**

1. **Распознавание составных слов по паттернам:**

   - Слова с дефисами: `Plug-in-Hybrid`, `Hotel-`
   - Немецкие составные слова без пробелов: `Sonnenuntergang`, `Lebensdauer`
   - Имена собственные: `Annemarie`, `Süderbrarup`, `Karsten`

2. **Создание словаря имён:**

   - Файл: `data/dictionaries/proper_nouns.dict`
   - Формат: список имён собственных (одно имя на строку)
   - Использование: для определения типа слова при выравнивании

3. **Реализация функции определения типа слова:**

В `modules/alignment.py` добавить:

```python
def get_word_type(word: str, proper_nouns: Optional[Set[str]] = None) -> str:
    """
    Определяет тип слова для применения специальных правил выравнивания.
    
    Args:
        word: Слово для анализа
        proper_nouns: Множество имён собственных (загружается из словаря)
    
    Returns:
        'compound': составное слово
        'proper_noun': имя собственное
        'regular': обычное слово
    """
    if proper_nouns is None:
        proper_nouns = load_proper_nouns_dict()
    
    # Проверка на составное слово
    if '-' in word:
        return 'compound'
    
    # Проверка на имя собственное
    if word in proper_nouns:
        return 'proper_noun'
    
    return 'regular'

def load_proper_nouns_dict() -> Set[str]:
    """Загружает словарь имён собственных из файла."""
    dict_path = PROJECT_ROOT / "data" / "dictionaries" / "proper_nouns.dict"
    if not dict_path.exists():
        return set()
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}
```

#### 1.2 Применение специальных правил при выравнивании

**Модификация `needleman_wunsch_align()` в `modules/alignment.py`:**

Добавить параметр для контекстных штрафов:

```python
def needleman_wunsch_align(
    sequence1: List[str],
    sequence2: List[str],
    match_score: float = 1.0,
    mismatch_score: float = -1.0,
    gap_penalty: float = -1.0,
    use_similarity_matrix: bool = True,
    word_type: str = 'regular'  # Новый параметр
) -> Tuple[List[Tuple[Optional[str], Optional[str]]], float]:
    """
    Perform Needleman-Wunsch global alignment.
    
    Args:
        word_type: Тип слова ('compound', 'proper_noun', 'regular')
                  Влияет на штрафы за пропуски
    """
    # Адаптация штрафов в зависимости от типа слова
    if word_type == 'compound':
        # Более мягкие штрафы для составных слов
        effective_gap_penalty = gap_penalty * 0.8
    elif word_type == 'proper_noun':
        # Средние штрафы для имён
        effective_gap_penalty = gap_penalty * 0.9
    else:
        effective_gap_penalty = gap_penalty
    
    # Использовать effective_gap_penalty вместо gap_penalty в алгоритме
    # ... остальной код ...
```

### Приоритет 3: Постобработка выравнивания

#### 3.1 Создание модуля постобработки

**Новый файл:** `modules/alignment_postprocessing.py`

**Функции:**

1. **Объединение соседних пропусков:**

   - Если несколько пропусков подряд в одной последовательности, возможно, это артефакт выравнивания
   - Функция: `merge_consecutive_gaps()`

2. **Валидация выравнивания:**

   - Проверка на нереалистичные паттерны (например, слишком много пропусков подряд)
   - Функция: `validate_alignment()`

3. **Сглаживание выравнивания:**

   - Для составных слов и длинных последовательностей
   - Функция: `smooth_alignment()`

**Пример структуры:**

```python
"""
Постобработка результатов выравнивания фонем.
"""

from typing import List, Tuple, Optional

def merge_consecutive_gaps(
    aligned_pairs: List[Tuple[Optional[str], Optional[str]]],
    max_consecutive: int = 3
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Объединяет соседние пропуски, если их слишком много подряд.
    
    Args:
        aligned_pairs: Результат выравнивания
        max_consecutive: Максимальное количество пропусков подряд
    
    Returns:
        Обработанные пары выравнивания
    """
    # Реализация
    pass

def validate_alignment(
    aligned_pairs: List[Tuple[Optional[str], Optional[str]]],
    max_gap_ratio: float = 0.3
) -> bool:
    """
    Проверяет выравнивание на нереалистичные паттерны.
    
    Args:
        aligned_pairs: Результат выравнивания
        max_gap_ratio: Максимальная доля пропусков от общей длины
    
    Returns:
        True если выравнивание валидно
    """
    # Реализация
    pass

def smooth_alignment(
    aligned_pairs: List[Tuple[Optional[str], Optional[str]]],
    word_type: str = 'regular'
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Сглаживает выравнивание для составных слов.
    
    Args:
        aligned_pairs: Результат выравнивания
        word_type: Тип слова
    
    Returns:
        Сглаженные пары выравнивания
    """
    # Реализация
    pass
```

#### 3.2 Интеграция постобработки

**Изменения в `app.py`:**

После вызова `needleman_wunsch_align()` добавить постобработку:

```python
from modules.alignment_postprocessing import smooth_alignment, validate_alignment

# После выравнивания
aligned_pairs, alignment_score = needleman_wunsch_align(...)

# Постобработка
word_type = get_word_type(text)  # Определить тип слова
aligned_pairs = smooth_alignment(aligned_pairs, word_type=word_type)

# Валидация
if not validate_alignment(aligned_pairs):
    # Логирование или обработка невалидного выравнивания
    pass
```

## Фаза 3: Долгосрочные улучшения (1-2 месяца)

### Приоритет 4: Улучшение обработки заимствований

#### 4.1 Создание словаря заимствований

**Файл:** `data/dictionaries/loanwords.dict`

**Формат:**

```
Computer	kʰɔmpjuːtɐ
Software	zɔftvɛːɐ
Internet	ɪntɐnɛt
```

**Использование:**

- Для улучшения транскрипции expected фонем (G2P)
- Для определения типа слова при выравнивании
- Для применения специальных правил схожести

#### 4.2 Адаптация параметров для заимствований

**Изменения в `modules/alignment.py`:**

Расширить функцию `get_word_type()`:

```python
def get_word_type(
    word: str,
    proper_nouns: Optional[Set[str]] = None,
    loanwords: Optional[Set[str]] = None
) -> str:
    """
    Определяет тип слова для применения специальных правил выравнивания.
    
    Returns:
        'compound': составное слово
        'proper_noun': имя собственное
        'loanword': заимствование
        'regular': обычное слово
    """
    if loanwords is None:
        loanwords = load_loanwords_dict()
    
    # ... существующий код ...
    
    if word.lower() in loanwords:
        return 'loanword'
    
    # ... остальной код ...
```

**Специальные правила схожести для заимствований:**

В `modules/phoneme_similarity.py` добавить правила для английских звуков:

```python
# Специальные правила для заимствований (английские звуки)
LOANWORD_SIMILARITY_RULES = {
    ('ʁ', 'ɹ'): 0.7,  # Немецкий увулярный R и английский альвеолярный R
    ('ŋ', 'n'): 0.6,  # Носовой согласный в разных позициях
    ('ɛ', 'ə'): 0.65,  # В заимствованиях часто путаются
}

# Использовать эти правила, если слово определено как заимствование
```

### Приоритет 5: Визуализация и отладка

#### 5.1 Улучшенная визуализация

**Файл:** `modules/visualization.py`

**Добавить:**

1. Выделение составных слов в визуализации
2. Показ структуры слова (составное/обычное/заимствование)
3. Статистику по типам ошибок (подстановки/удаления/вставки)

#### 5.2 Отладочный режим

**Добавить в `config.py`:**

```python
DEBUG_ALIGNMENT = False  # Включить для отладки выравнивания
```

**Добавить логирование в `modules/alignment.py`:**

```python
if config.DEBUG_ALIGNMENT:
    logger.debug(f"Alignment for word type: {word_type}")
    logger.debug(f"Gap penalty: {effective_gap_penalty}")
    logger.debug(f"Alignment score: {alignment_score}")
```

## Порядок реализации

### Фаза 2 (средний приоритет):

1. Создать словарь имён собственных
2. Реализовать функцию `get_word_type()`
3. Модифицировать `needleman_wunsch_align()` для поддержки типов слов
4. Создать модуль постобработки
5. Интегрировать постобработку в основной поток

### Фаза 3 (долгосрочные):

1. Создать словарь заимствований
2. Расширить правила схожести для заимствований
3. Улучшить визуализацию
4. Добавить отладочный режим
5. (Опционально) Автоматическое обучение правил на основе данных

## Ожидаемые результаты

### Фаза 2:

- Улучшение выравнивания для составных слов на 30-50%
- Снижение PER для имён собственных
- Более стабильные результаты выравнивания

### Фаза 3:

- Улучшение обработки заимствований (PER с 0.27 до 0.24-0.25)
- Лучшая визуализация для отладки
- Более гибкая система с возможностью расширения

## Файлы для изменения/создания

### Фаза 2:

1. `modules/alignment.py` - добавить `get_word_type()` и модифицировать `needleman_wunsch_align()`
2. `modules/g2p_module.py` - добавить логику определения составных слов
3. `modules/alignment_postprocessing.py` (создать) - модуль постобработки
4. `data/dictionaries/proper_nouns.dict` (создать) - словарь имён
5. `app.py` - интегрировать постобработку

### Фаза 3:

6. `data/dictionaries/loanwords.dict` (создать) - словарь заимствований
7. `modules/phoneme_similarity.py` - добавить правила для заимствований
8. `modules/visualization.py` - улучшить визуализацию
9. `config.py` - добавить `DEBUG_ALIGNMENT`

## Примечания

- Фаза 2 требует больше изменений в коде, чем Фаза 1
- Фаза 3 может выполняться постепенно, параллельно с использованием системы
- Все изменения должны быть протестированы на том же наборе из 1000 записей
- Рекомендуется делать изменения инкрементально и тестировать после каждого шага
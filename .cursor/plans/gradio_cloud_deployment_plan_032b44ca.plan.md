---
name: Gradio Cloud Deployment Plan
overview: План деплоя проекта SpeechRec-German-diagnostic с интеграцией german-phoneme-validator на Gradio Spaces (Hugging Face) для работы из облака.
todos:
  - id: setup_repo_structure
    content: "Создать структуру репозитория для Gradio Spaces: объединить оба проекта в один, скопировать german-phoneme-validator в основную директорию или настроить git submodule"
    status: pending
  - id: fix_paths
    content: "Исправить все пути: модуль phoneme_validator.py (путь к валидатору), config.py (относительные пути вместо абсолютных), app.py (примеры с абсолютными путями)"
    status: pending
    dependencies:
      - setup_repo_structure
  - id: merge_requirements
    content: Объединить requirements.txt из обоих проектов, разрешить конфликты версий (numpy<2.0, scikit-learn<1.4), добавить недостающие зависимости
    status: pending
  - id: setup_git_lfs
    content: "Настроить Git LFS для больших файлов: .gitattributes для .pt, .joblib, .pickle файлов в artifacts/ и data/dictionaries/"
    status: pending
    dependencies:
      - setup_repo_structure
  - id: create_packages_txt
    content: Создать packages.txt для установки системных зависимостей (espeak-ng, espeak-ng-data)
    status: pending
  - id: create_spaceignore
    content: Создать .spaceignore для исключения ненужных файлов (notebooks/, cursor_scripts/, __pycache__/, .mfa_temp/)
    status: pending
  - id: create_space_readme
    content: Создать README.md для Hugging Face Spaces с описанием проекта, примером использования, информацией о моделях
    status: pending
  - id: test_local_build
    content: "Протестировать локальную сборку: проверить импорты, загрузку моделей, работу G2P, корректность путей"
    status: pending
    dependencies:
      - fix_paths
      - merge_requirements
      - create_packages_txt
  - id: optimize_loading
    content: "Оптимизировать загрузку моделей: убедиться в lazy loading валидатора, добавить обработку ошибок при первом запуске, логирование прогресса"
    status: pending
    dependencies:
      - fix_paths
---

# План деплоя проекта на Gradio Spaces

## Анализ проекта

### Основной проект: SpeechRec-German-diagnostic

- **Frontend**: Gradio 4.0+ (UI для валидации произношения)
- **Основные модели**:
  - `facebook/wav2vec2-xlsr-53-espeak-cv-ft` - распознавание фонем (Hugging Face, скачивается автоматически)
  - Whisper `medium` - ASR (OpenAI, скачивается автоматически)
- **Словари** (локальные файлы):
  - `data/dictionaries/de_ipa.dsl` - IPA-Dict-DSL
  - `data/dictionaries/german_mfa.dict` - MFA Dictionary
  - Pickle кэши (`*.pickle`)
- **Системные зависимости**: eSpeak NG (для G2P)
- **Опционально**: MFA (отключен по умолчанию)

### Валидатор: german-phoneme-validator

- **22 обученные DL модели** в `artifacts/`:
  - По одной папке на пару фонем (например, `b-p_dl_models_with_context_v2/`)
  - В каждой: `best_model.pt`, `feature_scaler.joblib`, `feature_cols.json`
- **Интеграция**: через относительный путь `../german-phoneme-validator` (или копия в проект)

## Задачи деплоя

### 1. Структура репозитория для Gradio

Создать структуру для Hugging Face Spaces:

```
speechrec-german-diagnostic/
├── app.py                          # Основной файл (переименован или скопирован)
├── requirements.txt                # Объединенные зависимости
├── README.md
├── config.py
├── diagnostic_matrix.json
├── phoneme_normalization_table.json
├── modules/                        # Все модули из основного проекта
├── data/
│   └── dictionaries/              # Словари (git LFS для больших файлов)
├── german-phoneme-validator/      # Копия валидатора ИЛИ git submodule
│   ├── artifacts/                 # 22 модели (git LFS обязательно!)
│   └── core/
└── .spaceignore                   # Файлы для исключения из деплоя
```

### 2. Объединение зависимостей

Объединить `requirements.txt` из обоих проектов с учетом:

- Версии библиотек (numpy<2.0, scikit-learn<1.4 для валидатора)
- Разрешение конфликтов версий
- Системные зависимости (eSpeak NG через apt)

### 3. Адаптация путей

Изменить пути в `modules/phoneme_validator.py`:

- Вместо `Path(__file__).parent.parent.parent / "german-phoneme-validator"`
- Использовать `Path(__file__).parent.parent / "german-phoneme-validator"` (если копия)
- Или переменную окружения `VALIDATOR_PATH`

### 4. Системные зависимости

Создать `packages.txt` для Gradio:

```
espeak-ng
espeak-ng-data
```

### 5. Хранение больших файлов

- **Словари**: Git LFS для `.pickle` файлов
- **Модели валидатора**: Git LFS для всех `.pt`, `.joblib` файлов в `artifacts/`
- **Hugging Face модели**: Автоматически кэшируются при первом запуске

### 6. Оптимизация размера

- `.spaceignore` для исключения:
  - `notebooks/`
  - `cursor_scripts/`
  - `__pycache__/`
  - `.mfa_temp/`
  - Тестовые файлы

### 7. Переменные окружения

Настроить через Gradio Secrets (если нужно):

- `HF_TOKEN` (для приватных моделей)
- `VALIDATOR_PATH` (опционально)

### 8. Обработка кэша моделей

- Hugging Face модели кэшируются автоматически в `/root/.cache/huggingface/`
- Whisper модели кэшируются автоматически
- Первый запуск будет долгим (скачивание моделей)

## Реализация

### Файлы для создания/модификации

1. **app.py** - проверить пути, добавить обработку ошибок при первом запуске
2. **requirements.txt** - объединить зависимости из обоих проектов
3. **packages.txt** - системные пакеты (eSpeak NG)
4. **README.md** - описание для Hugging Face Spaces
5. **modules/phoneme_validator.py** - исправить пути к валидатору
6. **.gitattributes** - Git LFS для больших файлов
7. **.spaceignore** - исключить ненужные файлы
8. **Dockerfile** (опционально) - для кастомной конфигурации

### Критические изменения

1. **Путь к валидатору**: 

   - Изменить в `modules/phoneme_validator.py:13` с `parent.parent.parent` на `parent.parent`
   - Или использовать переменную окружения

2. **Относительные пути**:

   - Все абсолютные пути `/Volumes/SSanDisk/...` заменить на относительные
   - Проверить в `app.py`, `config.py`, примерах

3. **eSpeak NG**:

   - Убедиться, что устанавливается через `packages.txt`
   - Проверить работу G2P модуля

4. **Модели валидатора**:

   - Убедиться, что `artifacts/` копируется/подключается
   - Git LFS для `.pt`, `.joblib` файлов

## Проверка работоспособности

1. Первый запуск должен скачать Hugging Face модели
2. Валидатор должен загрузить 22 модели из `artifacts/`
3. G2P должен работать через eSpeak NG
4. Все модули должны импортироваться корректно
5. Тестовый запрос должен обработаться успешно

## Возможные проблемы

1. **Размер репозитория**: 

   - Модели валидатора ~500MB+ (нужен Git LFS)
   - Словари могут быть большими

2. **Время первого запуска**:

   - Скачивание моделей Hugging Face может занять 5-10 минут
   - Gradio Spaces имеет таймаут запуска (проверить)

3. **Память**:

   - Все модели в памяти могут потребовать CPU instance с достаточным RAM
   - Рассмотреть lazy loading для валидатора

4. **MFA**:

   - Не используется по умолчанию (MFA_ENABLED=False)
   - Если нужно включить, потребуется conda в Dockerfile
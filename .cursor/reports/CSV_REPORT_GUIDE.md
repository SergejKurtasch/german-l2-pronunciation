# Руководство по CSV отчетам

## 📋 Что было добавлено в notebook

В конец notebook `deep_phoneme_mismatch_analysis.ipynb` добавлена **Секция 13**: генерация детальных CSV отчетов.

---

## 🎯 Создаваемые файлы

После запуска notebook в папке `data/analysis_results/` будут созданы 3 CSV файла:

### 1. **phoneme_analysis_detailed_report.csv**
- **Размер**: ~1000 строк (все проанализированные предложения)
- **Назначение**: Основной детальный отчет

### 2. **phoneme_analysis_worst_100.csv**
- **Размер**: 100 строк
- **Назначение**: 100 предложений с худшим качеством (highest PER)

### 3. **phoneme_analysis_best_100.csv**
- **Размер**: 100 строк
- **Назначение**: 100 предложений с лучшим качеством (lowest PER)

---

## 📊 Структура CSV файла

| Колонка | Тип | Описание | Пример |
|---------|-----|----------|--------|
| `sentence` | str | Исходное предложение | "Das ist ein Test." |
| `correct_phonemes` | int | Количество правильных фонем | 15 |
| `incorrect_phonemes` | int | Количество неправильных фонем | 3 |
| `incorrect_phonemes_list` | str | Список ошибок (через запятую) | "a→ɑ, -e, +ə" |
| `per` | float | Phoneme Error Rate (0.0-1.0) | 0.167 |
| `has_loanword` | bool | Есть заимствованные слова | True |
| `duration_seconds` | float | Длительность аудио | 4.5 |
| `total_expected_phonemes` | int | Всего ожидаемых фонем | 18 |
| `total_recognized_phonemes` | int | Всего распознанных фонем | 17 |

---

## 🔍 Формат колонки `incorrect_phonemes_list`

Ошибки записываются в специальном формате:

### Типы ошибок:

1. **`a→ɑ`** - **Замена (Substitution)**
   - Ожидали фонему `a`
   - Распознали фонему `ɑ`
   - Наиболее частый тип ошибки

2. **`-e`** - **Пропуск (Deletion)**
   - Ожидали фонему `e`
   - Модель её не распознала
   - Знак минус `-` перед фонемой

3. **`+ə`** - **Вставка (Insertion)**
   - Лишняя фонема `ə` в распознанном
   - Модель "галлюцинирует" фонему
   - Знак плюс `+` перед фонемой

### Примеры:

```
"a→ɑ, -e, +ə"               # 3 ошибки: замена, пропуск, вставка
"ɛ→e, ɛ→e, ə→ɐ"             # 3 замены
"-ʔ, -ʔ, -ʔ"                # 3 пропуска (glottal stops)
"+ə, +ɐ"                     # 2 вставки
""                           # Нет ошибок (пустая строка)
```

---

## 💻 Использование в Python

### Загрузка и базовый анализ

```python
import pandas as pd

# Загрузить отчет
df = pd.read_csv('data/analysis_results/phoneme_analysis_detailed_report.csv')

# Показать статистику
print(f"Total sentences: {len(df)}")
print(f"Average PER: {df['per'].mean():.3f}")
print(f"Total correct: {df['correct_phonemes'].sum():,}")
print(f"Total incorrect: {df['incorrect_phonemes'].sum():,}")
```

### Фильтрация проблемных предложений

```python
# Предложения с более чем 10 ошибками
high_errors = df[df['incorrect_phonemes'] > 10]
print(high_errors[['sentence', 'incorrect_phonemes', 'per']])

# Предложения с PER > 0.5 (50% ошибок)
bad_per = df[df['per'] > 0.5]
print(bad_per[['sentence', 'per', 'incorrect_phonemes_list']])
```

### Сравнение заимствованных vs немецких слов

```python
# Группировка по has_loanword
comparison = df.groupby('has_loanword').agg({
    'per': 'mean',
    'correct_phonemes': 'sum',
    'incorrect_phonemes': 'sum'
})
comparison.index = ['Regular', 'Loanwords']
print(comparison)
```

### Анализ частых ошибок

```python
# Извлечь все ошибки
all_errors = []
for errors_str in df['incorrect_phonemes_list'].fillna(''):
    if errors_str:
        all_errors.extend([e.strip() for e in errors_str.split(',')])

# Подсчитать частоту
from collections import Counter
error_counts = Counter(all_errors)

# Топ-20 ошибок
print("Top 20 most frequent errors:")
for error, count in error_counts.most_common(20):
    print(f"  {count:4d}x  {error}")
```

### Анализ по типам ошибок

```python
# Разделить ошибки по типам
substitutions = [e for e in all_errors if '→' in e]
deletions = [e for e in all_errors if e.startswith('-')]
insertions = [e for e in all_errors if e.startswith('+')]

print(f"Substitutions: {len(substitutions)}")
print(f"Deletions: {len(deletions)}")
print(f"Insertions: {len(insertions)}")

# Топ-10 замен
from collections import Counter
print("\nTop 10 substitutions:")
for sub, count in Counter(substitutions).most_common(10):
    print(f"  {count:4d}x  {sub}")
```

---

## 📈 Использование в Excel / Google Sheets

### Открытие файла

1. Открыть Excel / Google Sheets
2. File → Import → CSV
3. Выбрать `phoneme_analysis_detailed_report.csv`
4. **Важно**: Убедиться, что кодировка **UTF-8** для корректного отображения IPA символов

### Полезные операции

#### Сортировка:
- По `per` (по убыванию) → худшие предложения
- По `incorrect_phonemes` (по убыванию) → больше всего ошибок

#### Фильтрация:
- `has_loanword = TRUE` → только заимствованные слова
- `incorrect_phonemes > 10` → много ошибок
- `per < 0.1` → хорошее качество

#### Pivot Table:
- Rows: `has_loanword`
- Values: Average of `per`, Sum of `incorrect_phonemes`

---

## 🎯 Практические примеры анализа

### 1. Найти предложения с проблемными фонемами

```python
# Найти все предложения, где была ошибка с фонемой 'ɛ'
problem_phoneme = 'ɛ'
df_problem = df[df['incorrect_phonemes_list'].str.contains(problem_phoneme, na=False)]
print(f"Sentences with errors involving '{problem_phoneme}': {len(df_problem)}")
print(df_problem[['sentence', 'incorrect_phonemes_list']].head())
```

### 2. Сравнить короткие vs длинные предложения

```python
# Добавить категорию по длине
df['length_category'] = pd.cut(
    df['total_expected_phonemes'], 
    bins=[0, 20, 50, 100, float('inf')],
    labels=['Short (<20)', 'Medium (20-50)', 'Long (50-100)', 'Very long (100+)']
)

# Сравнить PER по категориям
print(df.groupby('length_category')['per'].mean())
```

### 3. Временной анализ

```python
# PER vs длительность аудио
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['duration_seconds'], df['per'], alpha=0.5)
plt.xlabel('Audio Duration (seconds)')
plt.ylabel('PER')
plt.title('PER vs Audio Duration')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📁 Структура папки

```
data/analysis_results/
├── README.md                              # Документация
├── phoneme_analysis_detailed_report.csv   # Основной отчет (~1000 строк)
├── phoneme_analysis_worst_100.csv         # Худшие 100
└── phoneme_analysis_best_100.csv          # Лучшие 100
```

---

## ⚙️ Регенерация отчетов

Для создания новых отчетов:

1. Открыть `notebooks/deep_phoneme_mismatch_analysis.ipynb`
2. **Restart Kernel**
3. **Run All Cells**
4. Дождаться завершения (~15-30 минут)
5. Новые CSV файлы будут в `data/analysis_results/`

---

## ⚠️ Важные замечания

1. **Кодировка UTF-8**: Все файлы сохранены в UTF-8 для IPA символов
2. **Размер файлов**: Основной отчет ~1-2 MB (добавлен в `.gitignore`)
3. **Alignment**: Используется Needleman-Wunsch из `modules/alignment.py`
4. **PER**: Рассчитывается через `modules/metrics.py` (как в основном приложении)

---

## 🔗 Связь с другими отчетами

Эти CSV отчеты дополняют другие файлы из notebook:

- `data/phoneme_comparison_results.csv` - полные данные alignment
- `data/phoneme_analysis_summary.json` - сводная статистика
- `data/*.png` - визуализации confusion matrix, PER distribution, etc.

---

**Дата создания**: 2026-01-12  
**Автор**: Automated from `deep_phoneme_mismatch_analysis.ipynb`

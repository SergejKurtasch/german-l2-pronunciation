# 📊 Руководство по версионированию анализа

**Дата создания:** 2026-01-12  
**Ноутбук:** `notebooks/deep_phoneme_mismatch_analysis.ipynb`

---

## 🎯 Что это такое?

Система версионирования позволяет запускать анализ **несколько раз с разными случайными выборками** из 1000 предложений. Каждый запуск создаёт **отдельные отчёты**, которые не перезаписывают предыдущие результаты.

---

## ⚙️ Как использовать

### Шаг 1: Откройте ноутбук

Откройте файл:
```
notebooks/deep_phoneme_mismatch_analysis.ipynb
```

### Шаг 2: Найдите ячейку конфигурации

В самом начале ноутбука (после заголовка) есть ячейка:

```python
# =============================================================================
# ⚙️ ANALYSIS CONFIGURATION
# =============================================================================
# Change this value before each run to get a new random sample and separate reports

ANALYSIS_STAGE = 1  # 👈 CHANGE THIS BEFORE EACH RUN!
```

### Шаг 3: Измените `ANALYSIS_STAGE`

**Для первого запуска:**
```python
ANALYSIS_STAGE = 1  # ← Оставьте как есть
```

**Для второго запуска:**
```python
ANALYSIS_STAGE = 2  # ← Измените на 2
```

**Для третьего запуска:**
```python
ANALYSIS_STAGE = 3  # ← Измените на 3
```

И так далее...

### Шаг 4: Запустите ноутбук

В Jupyter:
```
Kernel → Restart & Run All
```

---

## 📁 Что создаётся

### Stage 1 (ANALYSIS_STAGE = 1):

```
data/
  ├── phoneme_comparison_results_stage1.csv
  └── phoneme_analysis_summary_stage1.json

data/analysis_results/
  ├── phoneme_analysis_detailed_report_stage1.csv
  ├── phoneme_analysis_worst_100_stage1.csv
  └── phoneme_analysis_best_100_stage1.csv
```

### Stage 2 (ANALYSIS_STAGE = 2):

```
data/
  ├── phoneme_comparison_results_stage2.csv
  └── phoneme_analysis_summary_stage2.json

data/analysis_results/
  ├── phoneme_analysis_detailed_report_stage2.csv
  ├── phoneme_analysis_worst_100_stage2.csv
  └── phoneme_analysis_best_100_stage2.csv
```

### Stage 3, 4, 5... и так далее

Каждый раз создаются новые файлы с суффиксом `_stageN`, где N — это номер стадии.

---

## 🎲 Случайная выборка

### Как это работает?

Каждая стадия использует **разный random seed**:

```python
ANALYSIS_STAGE = 1 → RANDOM_SEED = 100
ANALYSIS_STAGE = 2 → RANDOM_SEED = 200
ANALYSIS_STAGE = 3 → RANDOM_SEED = 300
```

Это гарантирует, что:
- ✅ Каждая стадия выбирает **разные 1000 предложений**
- ✅ Результаты **воспроизводимы** (если запустить Stage 1 снова, будут те же самые 1000 предложений)
- ✅ Можно **сравнивать** результаты между стадиями

### Сколько можно запустить стадий?

Теоретически — бесконечно много! Но практически:
- Всего в датасете **TV-2021.02-Neutral**: ~22,000 записей
- Каждая стадия выбирает: 1,000 записей
- Максимум уникальных стадий: ~22

После этого начнётся повторение (некоторые предложения будут в нескольких стадиях).

---

## 📊 Сравнение результатов между стадиями

### Пример: Сравнить PER между Stage 1 и Stage 2

```python
import pandas as pd

# Загрузить Stage 1
df_stage1 = pd.read_csv('data/analysis_results/phoneme_analysis_detailed_report_stage1.csv')
per_stage1 = df_stage1['per'].mean()

# Загрузить Stage 2
df_stage2 = pd.read_csv('data/analysis_results/phoneme_analysis_detailed_report_stage2.csv')
per_stage2 = df_stage2['per'].mean()

print(f"Stage 1 PER: {per_stage1:.3f}")
print(f"Stage 2 PER: {per_stage2:.3f}")
print(f"Difference: {abs(per_stage1 - per_stage2):.3f}")
```

### Пример: Объединить все стадии

```python
import pandas as pd
import glob

# Найти все отчёты
all_reports = glob.glob('data/analysis_results/phoneme_analysis_detailed_report_stage*.csv')

# Объединить
dfs = []
for report in all_reports:
    stage_num = int(report.split('_stage')[1].split('.')[0])
    df = pd.read_csv(report)
    df['stage'] = stage_num
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total records analyzed across all stages: {len(df_all)}")
print(f"\nPER by stage:")
print(df_all.groupby('stage')['per'].mean())
```

---

## 🔧 Технические детали

### Что изменяется между стадиями?

1. **Random seed** для выборки:
   ```python
   RANDOM_SEED = ANALYSIS_STAGE * 100
   ```

2. **Суффикс имён файлов**:
   ```python
   STAGE_SUFFIX = f"_stage{ANALYSIS_STAGE}"
   ```

3. **Используется в:**
   - `df_loanwords.sample(n=100, random_state=RANDOM_SEED)`
   - `df_regular.sample(n=900, random_state=RANDOM_SEED)`
   - `df_sample.sample(frac=1, random_state=RANDOM_SEED)`
   - Имена всех выходных файлов

### Что НЕ изменяется?

- ✅ Алгоритм анализа (одинаковый для всех стадий)
- ✅ Модель распознавания (та же модель)
- ✅ Параметры alignment (те же параметры)
- ✅ Датасет (тот же TV-2021.02-Neutral)

---

## 💡 Практические примеры использования

### Сценарий 1: Проверка стабильности модели

**Цель:** Убедиться, что PER модели стабилен на разных выборках.

```python
# Запустить 5 стадий (ANALYSIS_STAGE = 1, 2, 3, 4, 5)
# Сравнить средний PER:
# Stage 1: 0.235
# Stage 2: 0.242
# Stage 3: 0.238
# Stage 4: 0.240
# Stage 5: 0.237
# → Стандартное отклонение: 0.003 (модель стабильна!)
```

### Сценарий 2: Накопление большего датасета

**Цель:** Проанализировать больше предложений (например, 5000 вместо 1000).

```python
# Запустить ANALYSIS_STAGE = 1, 2, 3, 4, 5
# Объединить все отчёты → 5000 предложений
# Провести более глубокий статистический анализ
```

### Сценарий 3: A/B тестирование улучшений

**Цель:** Сравнить результаты до и после улучшения алгоритма.

```bash
# До улучшения (старая версия ноутбука):
ANALYSIS_STAGE = 1 → Stage 1 results

# После улучшения (новая версия ноутбука):
ANALYSIS_STAGE = 1 → Stage 1 results (с теми же 1000 предложениями!)

# Сравнить PER, confusion matrix, etc.
```

### Сценарий 4: Поиск outlier'ов

**Цель:** Найти предложения, которые стабильно плохо распознаются.

```python
# Запустить несколько стадий
# Найти предложения, которые попали в "worst 100" в нескольких стадиях
# Эти предложения — кандидаты на ручную проверку
```

---

## ⚠️ Важные замечания

### 1. Не забывайте изменять `ANALYSIS_STAGE`!

Если вы забудете изменить `ANALYSIS_STAGE` и запустите ноутбук снова:
- ✅ Результаты будут **воспроизводимы** (те же 1000 предложений)
- ❌ Старые файлы будут **перезаписаны**

**Решение:** Всегда проверяйте `ANALYSIS_STAGE` перед запуском!

### 2. Git и версионирование

Файлы отчётов **не должны** попадать в Git (они большие и часто меняются).

Убедитесь, что в `.gitignore` есть:
```
data/analysis_results/*.csv
data/phoneme_*.csv
data/phoneme_*.json
```

### 3. Дисковое пространство

Каждая стадия создаёт ~5-10 MB файлов.
- 10 стадий ≈ 50-100 MB
- 100 стадий ≈ 500 MB - 1 GB

Следите за дисковым пространством!

---

## 📋 Чек-лист для каждого нового запуска

- [ ] Открыть ноутбук `deep_phoneme_mismatch_analysis.ipynb`
- [ ] Найти ячейку с `ANALYSIS_STAGE`
- [ ] Изменить значение (например, с 1 на 2)
- [ ] Сохранить ноутбук (Ctrl+S / Cmd+S)
- [ ] Запустить `Kernel → Restart & Run All`
- [ ] Дождаться завершения (может занять 10-30 минут)
- [ ] Проверить, что создались новые файлы `*_stageN.csv`
- [ ] Проанализировать результаты

---

## 🆘 Troubleshooting

### Проблема: `NameError: name 'RANDOM_SEED' is not defined`

**Причина:** Вы запустили не все ячейки по порядку.

**Решение:**
```
Kernel → Restart & Run All
```

### Проблема: Файлы перезаписываются

**Причина:** Вы не изменили `ANALYSIS_STAGE` перед запуском.

**Решение:**
1. Измените `ANALYSIS_STAGE` на новое значение
2. Запустите снова

### Проблема: Не хватает места на диске

**Причина:** Накопилось много отчётов.

**Решение:**
```bash
# Удалить старые отчёты (ОСТОРОЖНО!)
rm data/analysis_results/phoneme_analysis_*_stage*.csv
rm data/phoneme_*_stage*.csv
rm data/phoneme_*_stage*.json

# Или архивировать:
tar -czf old_reports_$(date +%Y%m%d).tar.gz data/analysis_results/
```

---

## 📞 Дополнительная информация

### Связанные файлы:

- **Ноутбук:** `notebooks/deep_phoneme_mismatch_analysis.ipynb`
- **Модуль границ слов:** `modules/word_boundary_utils.py`
- **Отчёты:** `data/analysis_results/`

### Связанные руководства:

- `.cursor/reports/WORD_BOUNDARY_UPDATE_RU.md` - Улучшения границ слов
- `.cursor/reports/PHONEME_ALIGNMENT_ANALYSIS.md` - Анализ alignment
- `.cursor/reports/CSV_REPORT_GUIDE.md` - Структура CSV отчётов

---

**Автор:** Cursor AI Assistant  
**Проект:** SpeechRec-German-diagnostic  
**Версия:** 1.0

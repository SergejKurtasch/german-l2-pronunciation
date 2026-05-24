# 🚀 Быстрый старт: Версионирование анализа

## ✅ Готово!

Система версионирования **настроена и готова к использованию**.

---

## 📝 Как запустить новый анализ с другими 1000 предложениями?

### Шаг 1: Откройте ноутбук
```
notebooks/deep_phoneme_mismatch_analysis.ipynb
```

### Шаг 2: Измените `ANALYSIS_STAGE`

Найдите в самом начале ноутбука (ячейка 1):

```python
ANALYSIS_STAGE = 1  # 👈 ИЗМЕНИТЕ ЭТО ЧИСЛО!
```

**Измените на:**
- `ANALYSIS_STAGE = 2` для второго запуска
- `ANALYSIS_STAGE = 3` для третьего запуска
- И так далее...

### Шаг 3: Запустите ноутбук

```
Kernel → Restart & Run All
```

---

## 📁 Результаты

Каждый запуск создаст **отдельные файлы** с суффиксом `_stageN`:

### Stage 1 (первый запуск):
```
data/analysis_results/
  ├── phoneme_analysis_detailed_report_stage1.csv
  ├── phoneme_analysis_worst_100_stage1.csv
  └── phoneme_analysis_best_100_stage1.csv
```

### Stage 2 (второй запуск):
```
data/analysis_results/
  ├── phoneme_analysis_detailed_report_stage2.csv
  ├── phoneme_analysis_worst_100_stage2.csv
  └── phoneme_analysis_best_100_stage2.csv
```

---

## 🎲 Что изменяется?

- ✅ **Случайная выборка**: Каждый раз **разные 1000 предложений**
- ✅ **Имена файлов**: Каждый stage создаёт **новые файлы** (не перезаписывает старые)
- ✅ **Воспроизводимость**: Stage 1 всегда выберет одни и те же предложения

---

## 💡 Пример

```python
# Первый запуск
ANALYSIS_STAGE = 1
# → Файлы: *_stage1.csv, random_seed=100, предложения A-Z

# Второй запуск
ANALYSIS_STAGE = 2
# → Файлы: *_stage2.csv, random_seed=200, предложения AA-ZZ

# Третий запуск
ANALYSIS_STAGE = 3
# → Файлы: *_stage3.csv, random_seed=300, предложения AAA-ZZZ
```

---

## ⚠️ Важно!

**НЕ ЗАБУДЬТЕ изменить `ANALYSIS_STAGE` перед каждым новым запуском!**

Если вы забудете, старые файлы будут перезаписаны.

---

## 📚 Подробная документация

См. `.cursor/reports/ANALYSIS_VERSIONING_GUIDE.md`

---

**Готово! Можете запускать!** 🎉

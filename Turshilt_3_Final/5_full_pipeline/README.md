# 5. Full Pipeline + Тест

End-to-end pipeline + 574 unified test дээрх үнэлгээ.

## Файлууд

| Файл | Тайлбар |
|---|---|
| `t3_full_pipeline.ipynb` | **Гол notebook** — local end-to-end pipeline (Jupyter) |
| `build_unified_test.py` | 574 unified test CSV үүсгэх |
| `run_unified_test.py` | Pipeline inference (FAST/FULL mode) |
| `unified_test_predictions.csv` | 574 бүх зурагт хариу |
| `unified_test_summary.json` | Нэгтгэсэн метрик |

## End-to-end Pipeline logик

```python
def predict_full(image_path):
    img = read_image(image_path)
    
    # 1. Classification (3-fold + 6-view TTA)
    p_fracture = classify_image(img)            # 18 forward pass
    has_fracture = (p_fracture >= 0.2476)
    
    # 2-3. ЗӨВХӨН Cls=1 үед segmentation + severity
    if has_fracture:
        mask = predict_mask_tta(img)            # 4-view TTA U-Net
        severity, feats = predict_severity(mask)  # 10 feature → RF
        weeks_lo, weeks_hi, desc = recovery_weeks(severity)
        return {
            'fracture_prob': p_fracture,
            'has_fracture': True,
            'mask': mask,
            'severity': severity,
            'recovery_weeks': (weeks_lo, weeks_hi),
        }
    else:
        return {
            'fracture_prob': p_fracture,
            'has_fracture': False,
            # mask, severity, recovery байхгүй
        }
```

## 574 Unified Test үр дүн (FULL mode)

### Classification

| Метрик | Утга |
|---|---:|
| AUC | **0.9703** |
| F1 | 0.8037 |
| Accuracy | 0.9634 |
| Threshold | 0.2476 (CV-аас calibrate) |
| CM (TN, FP, FN, TP) | (510, 3, 18, 43) |

### Pipeline-conditional Segmentation

| Pipeline үр дагавар | Тоо |
|---|---:|
| Cls. "хагарал" → segm. ажилласан | 46 (43 TP + 3 FP) |
| Cls. "эрүүл" → segm. алгассан | 528 (510 TN + 18 FN) |

### TP=43 дээрх Segmentation үр дүн

| Метрик | Mean ± Std | Median |
|---|---|---|
| Dice | **0.4770 ± 0.3391** | 0.5779 |
| IoU | 0.3762 ± 0.2861 | 0.4064 |

## Хэрэглэх

```bash
# 1. CSV үүсгэх (анх удаа)
python build_unified_test.py

# 2. Pipeline-conditional inference (FAST: ~1 мин)
python run_unified_test.py

# 3. Бүрэн нарийвчлалтай (FULL: ~17 мин)
python run_unified_test.py --full
```

## Test set бэлдэлт

[data_unified_test.csv](../1_data_preparation/data_unified_test.csv):
- 61 хагарал (FracAtlas DatasetNinja test/) — polygon GT
- 513 эрүүл (not fractured-аас санамсаргүй sample, seed=42)
- НИЙТ = 574

## Notebook-ын зориулалт

`t3_full_pipeline.ipynb` нь **Jupyter дотроос интерактив ажиллахад зориулагдсан**:
- 6-panel dashboard зурах
- macOS native file picker-аар зураг сонгох
- FracAtlas test-аас 10 зураг дээр demo хийх

Скрипт `run_unified_test.py` нь **574 бүх дээр автомат** ажиллаж тоо гаргадаг.

## Inference хугацаа (Apple MPS дээр)

| Mode | Folds | TTA | Хугацаа |
|---|---:|---|---:|
| FAST | 1 | OFF | ~65 сек |
| FULL | 3 | 6-view cls + 4-view seg | **~1020 сек (17 мин)** |

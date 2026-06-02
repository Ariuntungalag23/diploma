# 2. Classification — ConvNeXt-Base

Bone fracture binary classification (хагарал/эрүүл) model.

## Файлууд

| Файл | Тайлбар |
|---|---|
| `config.py` | Бүх hyperparameter, path |
| `model.py` | FractureModel (ConvNeXt-Base + LayerNorm + Linear), EMA, LLD optimizer |
| `data.py` | PyTorch Dataset, 16-bit window/level, рентген augmentation |
| `train.py` | 3-fold StratifiedGroupKFold CV training |
| `infer.py` | TTA (320/384/448 × hflip) + 3-fold ensemble inference |
| `README.md` | Анхны README (тус кодын) |
| `checkpoints/fold0.pth, fold1.pth, fold2.pth` | EMA weights (symlink) |
| `checkpoints/cv_summary.json` | CV метрик + calibrated threshold |
| `checkpoints/oof.csv` | Out-of-fold prediction-ууд |

## Архитектур

```python
ConvNeXt-Base (timm: convnext_base.fb_in22k_ft_in1k_384)  ~88M params
  ↓ ImageNet-22K → 1K @ 384px pretrained
LayerNorm(feat_dim) + Dropout(0.2) + Linear(1)
  ↓
BCEWithLogitsLoss (pos_weight=5.78, label_smoothing=0.05)
```

## Сургалтын тохиргоо

| Hyperparameter | Утга |
|---|---|
| Cross-validation | 3-fold StratifiedGroupKFold (patient_id-аар) |
| Epoch (fold тус бүрт) | 12 |
| Batch size | 32 × accumulate 2 = effective 64 |
| Optimizer | AdamW + Layer-wise LR Decay (layer_decay=0.75) |
| LR (head / backbone) | 5e-4 / 5e-5 |
| EMA decay | 0.9995 |
| Mixed precision | FP16 + GradScaler |
| Mixup / CutMix | α=0.2 / α=1.0, 50% prob |
| Hardware | NVIDIA A6000 48GB, ~9 цаг |

## TTA Inference

- 3 scale × hflip (on/off) = 6 view per fold
- 3 fold × 6 view = 18 forward pass / image
- Multi-fold ensemble averaging

## Threshold calibration

`recall ≥ 0.95`-ийг хангах хамгийн дээд precision threshold-ийг OOF дээр
тооцоолсон → **0.2476**.

## Үр дүн ([checkpoints/cv_summary.json](checkpoints/cv_summary.json))

| Метрик | CV (33K train OOF) | Test (574 hold-out) |
|---|---:|---:|
| AUC | 0.9887 | **0.9703** |
| F1 | 0.959 | 0.804 |
| Recall (target) | 0.950 | — |
| Precision | 0.968 | — |

Generalization gap (CV - Test) = 0.018 — overfit байхгүй.

## Хэрэглэх

### Inference (data_test.csv дээр):
```bash
python infer.py
# → checkpoints/test_predictions.csv
# → checkpoints/test_summary.json
```

### 574 unified test:
```bash
# 5_full_pipeline/-аас:
python ../5_full_pipeline/run_unified_test.py --full
```

## Эх кодын байршил

Анхны: `/Users/ariuntungalag/Desktop/LAST/T3_classification/`

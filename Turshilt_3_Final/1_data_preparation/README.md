# 1. Өгөгдөл бэлдэх (Data Preparation)

## Файлууд

| Файл | Тайлбар |
|---|---|
| `build_dataset.py` | 4 өөр dataset нэгтгэх + 85/15 patient-level split |
| `data_train.csv` | 33,372 train (4 dataset нийтлэг) |
| `data_test.csv` | 5,768 hold-out test (group_id-аар ялгасан, group overlap = 0) |
| `data_unified_test.csv` | **574 нэгдсэн test** (61 fractured + 513 not_fractured, FracAtlas DatasetNinja-аас) |

## CSV-ийн баганууд

```
image_path  — зургийн бүтэн зам
label       — 0 (хагарал байхгүй) эсвэл 1 (хагарал байна)
group_id    — patient-level split-д ашиглана (leakage сэргийлэх)
dataset     — graz / fracatlas / bonefracturecv / multiregion
```

## 4 dataset

| # | Dataset | Train | Test | Pos ratio |
|---|---|---:|---:|---:|
| 1 | GRAZPEDWRI-DX (16-bit) | 17,341 | 2,986 | 67% / 65% |
| 2 | Multi-region X-ray | 9,028 | 1,553 | 49% / 51% |
| 3 | BoneFractureCV | 3,519 | 629 | 50% / 49% |
| 4 | FracAtlas | 3,483 | 600 | 18% / 18% |
| | **НИЙТ** | **33,372** | **5,768** | |

## Split логик ([build_dataset.py:561-577](build_dataset.py))

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(gss.split(df, df['label'], df['group_id']))
# Group overlap (Train ↔ Test): 0  ← Patient-level disjoint баталгаа
```

## Unified test (574) — segmentation-д хэрэглэх

Сегментаци-д polygon GT хэрэгтэй учир зөвхөн FracAtlas-ын зургийг
ашигладаг. Скрипт нь `5_full_pipeline/build_unified_test.py`-д байгаа.

```python
# 61 fractured + 513 not_fractured-аас санамсаргүй sample = 574
random.seed(42)
sampled_neg = random.sample(notfx_imgs, 513)
```

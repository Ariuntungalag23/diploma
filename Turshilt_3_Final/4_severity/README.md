# 4. Severity — Хүндрэлийн зэрэг ангилал

10 морфологийн feature дээр суурилсан RandomForest classifier.

## Файлууд

| Файл | Тайлбар |
|---|---|
| `severity_classifier_training.ipynb` | Бүх pipeline: feature extraction → RF training |
| `severity_classifier.pkl` | Хадгалсан model (symlink) |

## Pipeline тойм

```
Binary mask (segmentation-аас гарсан)
   ↓
10 морфологийн feature гаргах (skimage.measure.regionprops)
   ↓
Heuristic дүрмээр severity label үүсгэх (TRAIN-д)
   ↓
RandomForest classifier (200 trees)
   ↓
Inference: mask → 10 feature → grade 0/1/2/3
```

## 10 морфологийн feature

| # | Feature | Тайлбар |
|---|---|---|
| 1 | `area_px` | Маскын pixel тоо |
| 2 | `area_pct` | Зургийн талбайн хувь |
| 3 | `n_components` | Холбоост хэсгийн тоо |
| 4 | `largest_comp_pct` | Хамгийн том холбоост хэсгийн хувь |
| 5 | `major_axis` | Хагарлын урт (ellipse fit) |
| 6 | `minor_axis` | Хагарлын өргөн |
| 7 | `aspect_ratio` | major / minor |
| 8 | `eccentricity` | Тойргоос хэр зайлсан |
| 9 | `solidity` | area / convex_hull_area |
| 10 | `bone_region` | Centroid-ийн босоо хувь |

## Heuristic дүрэм (TRAIN labels)

```python
if area_px < 30:                              → grade 0  (Normal)
elif area_pct > 5% or n_comp >= 3 or solidity < 0.65:
                                              → grade 3  (Severe)
elif area_pct > 1% or n_comp == 2:           → grade 2  (Moderate)
else:                                         → grade 1  (Mild)
```

## RandomForest тохиргоо

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
# 574 train sample (FracAtlas DatasetNinja train+val+test polygon)
# 5-fold CV
```

## Recovery weeks lookup

```python
RECOVERY_WEEKS = {
    0: (0,  0,  'Эмчилгээ шаардлагагүй'),
    1: (2,  4,  'Бага зэргийн хугарал'),
    2: (6,  8,  'Дунд зэргийн хугарал'),
    3: (10, 16, 'Хүнд хугарал (мэс засал хэрэгтэй магадгүй)'),
}
```

## Хязгаарлалт

**Heuristic шошго** — радиологичын жинхэнэ severity grading биш. FracAtlas
dataset-д severity label байхгүй учир маскнаас тооцоолсон.

→ Дипломын ил хязгаарлалт болж бичигдсэн.

## Эх байршил

`/Users/ariuntungalag/Desktop/LAST/severity_classifier.pkl`

Хадгалсан хэлбэр:
```python
{
    'classifier': RandomForestClassifier,
    'feature_cols': [10 нэр],
    'cv_accuracy': float,
}
```

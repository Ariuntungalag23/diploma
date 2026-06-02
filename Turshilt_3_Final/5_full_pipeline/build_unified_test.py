"""
574 unified test CSV бэлдэх:
  61 fractured (FracAtlas DatasetNinja test) + 513 not_fractured (sample)

Гарах файл: T3_classification/data_unified_test.csv
  Багана: image_path, ann_path, label
    image_path : зургийн зам
    ann_path   : Supervisely polygon JSON (positives), эсвэл ""
    label      : 1 (fractured) эсвэл 0 (not fractured)
"""

import csv
import random
from pathlib import Path

LAST_ROOT = Path('/Users/ariuntungalag/Desktop/LAST')
FA = LAST_ROOT / 'datasets' / 'fracatlas-DatasetNinja'
OUT_CSV = LAST_ROOT / 'T3_classification' / 'data_unified_test.csv'
SEED = 42

random.seed(SEED)

# --- Fractured ---
fractured_imgs = sorted((FA / 'test' / 'img').glob('*.jpg'))
print(f'Fractured images: {len(fractured_imgs)}')

# --- Not fractured (sample) ---
notfx_imgs = sorted((FA / 'not fractured' / 'img').glob('*.jpg'))
print(f'Not_fractured available: {len(notfx_imgs)}')

N_NEG = 513  # 61 + 513 = 574
sampled_neg = random.sample(notfx_imgs, N_NEG)

# --- Build CSV ---
rows = []
for p in fractured_imgs:
    ann = FA / 'test' / 'ann' / (p.name + '.json')
    rows.append({
        'image_path': str(p),
        'ann_path': str(ann) if ann.exists() else '',
        'label': 1,
    })
for p in sampled_neg:
    ann = FA / 'not fractured' / 'ann' / (p.name + '.json')
    rows.append({
        'image_path': str(p),
        'ann_path': str(ann) if ann.exists() else '',
        'label': 0,
    })

random.shuffle(rows)

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['image_path', 'ann_path', 'label'])
    w.writeheader()
    w.writerows(rows)

print(f'\n  ✓ Хадгаласан: {OUT_CSV}')
print(f'    Нийт: {len(rows)} зураг')
print(f'    Pos (fractured)    : {sum(1 for r in rows if r["label"]==1)}')
print(f'    Neg (not fractured): {sum(1 for r in rows if r["label"]==0)}')
print(f'    Polygon GT-той     : {sum(1 for r in rows if r["ann_path"])}')

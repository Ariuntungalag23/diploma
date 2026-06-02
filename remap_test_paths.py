"""
data_test.csv-ийн зам /dev/shm/... → локал /Users/.../LAST/datasets/... болгож remap хийнэ.

Гаргадаг:
    T3_classification/data_test_local.csv     # remapped зам, оршихыг шалгасан
    T3_classification/data_test_missing.csv   # remap чадаагүй зургуудын жагсаалт
"""

import csv
import os
from pathlib import Path
from collections import Counter

LAST_ROOT  = Path('/Users/ariuntungalag/Desktop/LAST')
DATASETS   = LAST_ROOT / 'datasets'
TEST_CSV   = LAST_ROOT / 'T3_classification' / 'data_test.csv'
OUT_CSV    = LAST_ROOT / 'T3_classification' / 'data_test_local.csv'
MISS_CSV   = LAST_ROOT / 'T3_classification' / 'data_test_missing.csv'


def build_index():
    """Бүх локал зургуудын basename → бүтэн зам индекс."""
    print('Локал зургуудын индекс үүсгэж байна...')
    idx = {}

    # 1) GRAZPEDWRI-DX
    graz_root = DATASETS / 'GRAZPEDWRI-DX'
    n0 = 0
    for p in graz_root.rglob('*.png'):
        idx.setdefault(p.name, str(p))
        n0 += 1
    print(f'  GRAZPEDWRI-DX  : {n0:>6} PNG')

    # 2) FracAtlas (DatasetNinja локал)
    fa_root = LAST_ROOT / 'fracatlas-DatasetNinja'
    n1 = 0
    for split in ['train', 'val', 'test', 'not fractured']:
        d = fa_root / split / 'img'
        if d.exists():
            for p in d.glob('*.jpg'):
                idx.setdefault(p.name, str(p))
                n1 += 1
    print(f'  FracAtlas      : {n1:>6} JPG')

    # 3) BoneFractureCV (YOLOv8 structure)
    bfcv_root = DATASETS / 'BoneFractureCV'
    n2 = 0
    for p in bfcv_root.rglob('*.jpg'):
        if '/images/' in str(p):
            idx.setdefault(p.name, str(p))
            n2 += 1
    print(f'  BoneFractureCV : {n2:>6} JPG')

    # 4) Multi-region (folder/class structure)
    mr_root = DATASETS / 'MultiRegionXray'
    n3 = 0
    for p in mr_root.rglob('*.jpg'):
        idx.setdefault(p.name, str(p))
        n3 += 1
    for p in mr_root.rglob('*.png'):
        idx.setdefault(p.name, str(p))
        n3 += 1
    print(f'  MultiRegion    : {n3:>6} img')

    print(f'\nНийт уникал basename: {len(idx):,}')
    return idx


def main():
    if not TEST_CSV.exists():
        raise FileNotFoundError(TEST_CSV)

    idx = build_index()

    found_rows = []
    missing_rows = []
    found_by_ds = Counter()
    missing_by_ds = Counter()

    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        for row in reader:
            orig = row['image_path']
            base = os.path.basename(orig)
            ds = row.get('dataset', '?')
            local = idx.get(base)
            if local:
                row['image_path'] = local
                found_rows.append(row)
                found_by_ds[ds] += 1
            else:
                missing_rows.append(row)
                missing_by_ds[ds] += 1

    # Save outputs
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(found_rows)

    with open(MISS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(missing_rows)

    print(f'\n  ✓ Олдсон    : {len(found_rows):>5,} зураг → {OUT_CSV}')
    print(f'  ✗ Олдоогүй  : {len(missing_rows):>5,} зураг → {MISS_CSV}')
    print(f'\n  Олдсон зургуудыг dataset-аар:')
    for ds, n in sorted(found_by_ds.items()):
        miss = missing_by_ds.get(ds, 0)
        total = n + miss
        print(f'    {ds:20s}: {n:>5}/{total:<5}  ({n/total*100:.1f}%)')

    if missing_rows:
        print(f'\n  Олдоогүй жишээ (эхний 5):')
        for r in missing_rows[:5]:
            print(f'    [{r.get("dataset","?")}] {r["image_path"]}')


if __name__ == '__main__':
    main()

"""
================================================================================
BUILD DATASET — 4 dataset-ыг нэгтгэж 2 CSV файл үүсгэнэ:
    1. data_train.csv  — 85% (3-fold CV дотор эргэлдэнэ)
    2. data_test.csv   — 15% (хэзээ ч сургалтанд орохгүй)
================================================================================
CSV-ийн баганууд:
    image_path  — зургийн бүрэн зам
    label       — 0 (fracture байхгүй) эсвэл 1 (fracture байна)
    group_id    — patient-level split-д ашиглана (leakage урьдчилан сэргийлэх)
    dataset     — аль dataset-аас ирсэн (graz/fracatlas/bonefracturecv/yolobonefracture)
================================================================================

ҮНДСЭН ӨӨРЧЛӨЛТҮҮД (шинэ Kaggle GRAZPEDWRI-DX дэмжих):
  • images_part1/..images_part4/ структурыг дэмжинэ
  • /dev/shm/GRAZPEDWRI-DX-аас уншиж болно (хурдан RAM)
  • Зургуудыг нэг удаа index хийгээд O(1)-аар lookup хийдэг
================================================================================
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================================
# 1) GRAZPEDWRI-DX
# ============================================================================

def build_grazpedwri(root: Path) -> pd.DataFrame:
    """
    GRAZPEDWRI-DX dataset → DataFrame
    
    Дэмжих бүтэц:
      (A) Kaggle (шинэ) — олон хавтаст:
          root/
            dataset.csv
            images_part1/*.png
            images_part2/*.png
            images_part3/*.png
            images_part4/*.png
            folder_structure/
      
      (B) Гар татсан (хуучин) — нэг хавтаст:
          root/
            dataset.csv
            images/*.png
    
    Label логик:
      fracture_visible == 1 → label = 1 (хатуу хувилбар)
      бусад бүх tag       → label = 0
    """
    print(f"\n[1/4] GRAZPEDWRI-DX боловсруулж байна: {root}")
    
    csv_path = root / 'dataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv олдсонгүй: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  → dataset.csv: {len(df)} мөр, баганууд: {list(df.columns)[:10]}...")
    
    # Label багана: 'fracture_visible' эсвэл 'fracture' хайх
    label_col = None
    for cand in ['fracture_visible', 'fracture', 'has_fracture']:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(
            f"Fracture label багана олдсонгүй. Боломжит баганууд: {list(df.columns)}"
        )
    print(f"  → Label багана ашиглаж байна: '{label_col}'")
    
    # Patient_id багана хайх
    patient_col = None
    for cand in ['patient_id', 'patientid', 'patient']:
        if cand in df.columns:
            patient_col = cand
            break
    
    # filestem багана
    if 'filestem' not in df.columns:
        raise KeyError(f"'filestem' багана байхгүй: {list(df.columns)}")
    
    # ------------------------------------------------------------------------
    # Зургийн хавтсуудыг олох (ШИНЭ: images_part*/ дэмжих)
    # ------------------------------------------------------------------------
    images_dirs = []
    
    # (A) Шинэ Kaggle бүтэц — images_part1, images_part2, ...
    for part_dir in sorted(root.glob('images_part*')):
        if part_dir.is_dir():
            images_dirs.append(part_dir)
    
    # (B) Хуучин бүтэц — нэг images/ хавтас
    if not images_dirs and (root / 'images').exists():
        images_dirs.append(root / 'images')
    
    # (C) Бусад магадлалтай зам
    if not images_dirs:
        for alt in ['data/images', 'images/full', 'yolov5/images']:
            if (root / alt).exists():
                images_dirs.append(root / alt)
                break
    
    if not images_dirs:
        raise FileNotFoundError(
            f"images/, images_part*/, эсвэл бусад зургийн хавтас олдсонгүй: {root}\n"
            f"Гар шалгах: ls {root}"
        )
    
    print(f"  → Олдсон зургийн хавтсууд: {[d.name for d in images_dirs]}")
    
    # ------------------------------------------------------------------------
    # ШИНЭ: filename → path index үүсгэх (нэг удаа scan, O(1) lookup)
    # ------------------------------------------------------------------------
    print(f"  → Зургуудыг indexлэж байна...")
    file_index = {}
    duplicates = 0
    for d in images_dirs:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == '.png':
                if p.stem in file_index:
                    duplicates += 1
                file_index[p.stem] = p
    
    print(f"  → Нийт {len(file_index):,} өвөрмөц зураг indexлэгдсэн"
          + (f" (⚠ {duplicates} давхардал)" if duplicates else ""))
    
    # ------------------------------------------------------------------------
    # CSV-ийн мөр бүрийг index-ээс хайх
    # ------------------------------------------------------------------------
    rows = []
    missing = 0
    for _, r in df.iterrows():
        img_path = file_index.get(r['filestem'])
        if img_path is None:
            missing += 1
            continue
        
        # Patient_id: CSV-д байвал ашиглах, эс бөгөөс filename-аас задлах
        if patient_col:
            pid = str(r[patient_col])
        else:
            # Жишээ: "0001_1297860395_01_WRI-L1_M014" → "0001" нь patient_id
            pid = str(r['filestem']).split('_')[0]
        
        label = int(r[label_col] > 0) if not pd.isna(r[label_col]) else 0
        
        rows.append({
            'image_path': str(img_path.resolve()),
            'label': label,
            'group_id': f"graz_{pid}",
            'dataset': 'graz',
        })
    
    if missing > 0:
        print(f"  ⚠ {missing} зураг файл олдсонгүй (CSV-д бий, диск дээр алга)")
    
    out = pd.DataFrame(rows)
    pos = (out['label'] == 1).sum()
    print(f"  ✓ GRAZPEDWRI-DX: {len(out):,} зураг "
          f"({pos:,} fracture, {len(out)-pos:,} non-fracture, "
          f"{out['group_id'].nunique():,} unique patient)")
    return out


# ============================================================================
# 2) FracAtlas
# ============================================================================

def build_fracatlas(root: Path) -> pd.DataFrame:
    """
    FracAtlas dataset → DataFrame
    
    Бүтэц:
      root/
        dataset.csv           ← 'image_id' + 'fractured' + 'multiscan' гэх мэт
        images/
          Fractured/*.jpg
          Non_fractured/*.jpg
    
    Label логик:
      CSV-ийн 'fractured' багана = 0/1 шууд ашиглана
    """
    print(f"\n[2/4] FracAtlas боловсруулж байна: {root}")
    
    csv_path = root / 'dataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv олдсонгүй: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"  → dataset.csv: {len(df)} мөр, баганууд: {list(df.columns)[:10]}...")
    
    # 'fractured' багана хайх
    if 'fractured' not in df.columns:
        raise KeyError(f"'fractured' багана байхгүй: {list(df.columns)}")
    
    # 'image_id' эсвэл 'image' багана
    img_col = None
    for cand in ['image_id', 'image', 'filename', 'file_name']:
        if cand in df.columns:
            img_col = cand
            break
    if img_col is None:
        raise KeyError(f"Image багана олдсонгүй: {list(df.columns)}")
    
    img_root_candidates = [
        root / 'images',
        root / 'data' / 'images',
    ]
    images_root = None
    for c in img_root_candidates:
        if c.exists():
            images_root = c
            break
    if images_root is None:
        raise FileNotFoundError(f"images хавтас олдсонгүй: {root}")
    
    fractured_dir = None
    non_fractured_dir = None
    for sub in images_root.iterdir():
        if sub.is_dir():
            n = sub.name.lower().replace('-', '_')
            if 'non' in n and 'fractur' in n:
                non_fractured_dir = sub
            elif 'fractur' in n:
                fractured_dir = sub
    
    if fractured_dir is None or non_fractured_dir is None:
        raise FileNotFoundError(
            f"Fractured/Non_fractured хавтсууд олдсонгүй: {images_root}"
        )
    print(f"  → Fractured хавтас: {fractured_dir}")
    print(f"  → Non_fractured хавтас: {non_fractured_dir}")
    
    rows = []
    missing = 0
    for _, r in df.iterrows():
        img_name = r[img_col]
        label = int(r['fractured'])
        
        if not str(img_name).lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name = str(img_name) + '.jpg'
        
        if label == 1:
            img_path = fractured_dir / img_name
        else:
            img_path = non_fractured_dir / img_name
        
        if not img_path.exists():
            alt = non_fractured_dir / img_name if label == 1 else fractured_dir / img_name
            if alt.exists():
                img_path = alt
            else:
                missing += 1
                continue
        
        rows.append({
            'image_path': str(img_path.resolve()),
            'label': label,
            'group_id': f"fracatlas_{Path(img_name).stem}",
            'dataset': 'fracatlas',
        })
    
    if missing > 0:
        print(f"  ⚠ {missing} зураг файл олдсонгүй")
    
    out = pd.DataFrame(rows)
    pos = (out['label'] == 1).sum()
    print(f"  ✓ FracAtlas: {len(out):,} зураг "
          f"({pos:,} fracture, {len(out)-pos:,} non-fracture)")
    return out


# ============================================================================
# 3) BoneFractureCV — Kaggle (pkdarabi, Roboflow толин хувилбар), YOLO формат
# ============================================================================

ROBOFLOW_AUG_PATTERN = re.compile(r'\.rf\.[0-9a-f]{16,}\.', re.IGNORECASE)


def build_yolo_dataset(root: Path, name: str, exclude_augmented: bool = True) -> pd.DataFrame:
    """
    YOLO формат dataset → DataFrame (pkdarabi Kaggle хувилбар)
    """
    print(f"\n[{name}] YOLO dataset боловсруулж байна: {root}")
    
    if not root.exists():
        raise FileNotFoundError(f"Хавтас олдсонгүй: {root}")
    
    rows = []
    augmented_skipped = 0
    
    for split in ['train', 'valid', 'val', 'test']:
        img_dir = root / split / 'images'
        lbl_dir = root / split / 'labels'
        
        if not img_dir.exists():
            continue
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            
            if exclude_augmented and ROBOFLOW_AUG_PATTERN.search(img_path.name):
                augmented_skipped += 1
                continue
            
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                label = 0
            else:
                with open(lbl_path) as f:
                    content = f.read().strip()
                label = 1 if content else 0
            
            rows.append({
                'image_path': str(img_path.resolve()),
                'label': label,
                'group_id': f"{name}_{img_path.stem}",
                'dataset': name,
            })
    
    if augmented_skipped > 0:
        print(f"  → Augmented хувилбарууд хасагдсан: {augmented_skipped}")
    
    out = pd.DataFrame(rows)
    
    # Empty DataFrame шалгалт
    if len(out) == 0:
        print(f"  ⚠ {name}: 0 зураг олдсон!")
        if augmented_skipped > 0:
            print(f"  💡 БҮХ {augmented_skipped} файл нь Roboflow augmented байсан.")
            print(f"     --include_roboflow_augmented flag ашиглан дахин ажиллуул.")
        return pd.DataFrame(columns=['image_path', 'label', 'group_id', 'dataset'])
    
    pos = (out['label'] == 1).sum()
    print(f"  ✓ {name}: {len(out):,} зураг "
          f"({pos:,} fracture, {len(out)-pos:,} non-fracture)")
    return out


# ============================================================================
# 4) MultiRegionXray — Kaggle (bmadushanirodrigo), folder-based classification
# ============================================================================

def build_multiregion(root: Path) -> pd.DataFrame:
    """
    Bone Fracture Multi-Region X-ray dataset → DataFrame
    """
    print(f"\n[4/4] MultiRegionXray боловсруулж байна: {root}")
    
    if not root.exists():
        raise FileNotFoundError(f"Хавтас олдсонгүй: {root}")
    
    base_dir = root
    if not (root / 'train').exists():
        found = False
        for sub in root.rglob('train'):
            if sub.is_dir() and ((sub.parent / 'val').exists()
                                  or (sub.parent / 'test').exists()
                                  or (sub.parent / 'valid').exists()):
                base_dir = sub.parent
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"train/ хавтас олдсонгүй: {root}\n"
                f"Гар шалгах: find {root} -maxdepth 3 -type d"
            )
    print(f"  → Base directory: {base_dir}")
    
    rows = []
    
    for split in ['train', 'val', 'valid', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        
        for sub in split_dir.iterdir():
            if not sub.is_dir():
                continue
            
            name_norm = sub.name.lower().replace('_', ' ').replace('-', ' ')
            if 'not' in name_norm and 'fractur' in name_norm:
                label = 0
            elif 'fractur' in name_norm:
                label = 1
            else:
                print(f"  ⚠ Тодорхойгүй хавтас алгаслаа: {sub.name}")
                continue
            
            for img_path in sub.iterdir():
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                rows.append({
                    'image_path': str(img_path.resolve()),
                    'label': label,
                    'group_id': f"multiregion_{img_path.stem}",
                    'dataset': 'multiregion',
                })
    
    out = pd.DataFrame(rows)
    pos = (out['label'] == 1).sum()
    print(f"  ✓ MultiRegionXray: {len(out):,} зураг "
          f"({pos:,} fracture, {len(out)-pos:,} non-fracture)")
    return out


# ============================================================================
# MAIN — Бүгдийг нэгтгэж train/test split хийх
# ============================================================================

def resolve_nested(root: Path, markers: list) -> Path:
    """
    Kaggle dataset-уудад nested folder бий болох нь түгээмэл:
        FracAtlas/
          FracAtlas/           ← давхар хавтас
            dataset.csv
            images/
    
    Энэ функц markers-ийн аль нэгийг олох хүртэл 1 түвшин гүн рүү ордог.
    
    Жишээ:
        resolve_nested(Path('/workspace/datasets/FracAtlas'),
                       ['dataset.csv', 'images'])
        → /workspace/datasets/FracAtlas/FracAtlas
    """
    # 1) root дотроос marker олох
    for m in markers:
        if (root / m).exists():
            return root
    
    # 2) ижил нэртэй nested folder шалгах (FracAtlas/FracAtlas/)
    nested_same = root / root.name
    if nested_same.is_dir():
        for m in markers:
            if (nested_same / m).exists():
                print(f"  → Nested folder илрэв: {nested_same}")
                return nested_same
    
    # 3) Цорын ганц subdirectory байгаа эсэхийг шалгах
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        for m in markers:
            if (subdirs[0] / m).exists():
                print(f"  → Nested folder илрэв: {subdirs[0]}")
                return subdirs[0]
    
    # Олдсонгүй — root-ийг буцаах (build_* функц өөрөө алдаа гаргана)
    return root


def resolve_graz_root(args_graz_root: str, root: Path) -> Path:
    """
    GRAZPEDWRI-DX-ийн root хавтсыг олох логик:
    
    1. --graz_root заасан бол түүнийг ашиглах
    2. /dev/shm/GRAZPEDWRI-DX (хурдан RAM) байвал ашиглах
    3. {--root}/GRAZPEDWRI-DX-ыг ашиглах (анхдагч)
    """
    if args_graz_root:
        return Path(args_graz_root).resolve()
    
    shm_path = Path('/dev/shm/GRAZPEDWRI-DX')
    if shm_path.exists() and (shm_path / 'dataset.csv').exists():
        print(f"  → /dev/shm-ээс GRAZPEDWRI-DX ашиглаж байна (хурдан RAM): {shm_path}")
        return shm_path
    
    return root / 'GRAZPEDWRI-DX'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/workspace/datasets',
                        help='Dataset-уудын root хавтас (default: /workspace/datasets)')
    parser.add_argument('--graz_root', type=str, default=None,
                        help='GRAZPEDWRI-DX-ийн root зам. '
                             'Default: /dev/shm/GRAZPEDWRI-DX байвал тэр, '
                             'үгүй бол {--root}/GRAZPEDWRI-DX')
    parser.add_argument('--train_csv', type=str, default='./data_train.csv')
    parser.add_argument('--test_csv', type=str, default='./data_test.csv')
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_roboflow_augmented', action='store_true',
                        help='Roboflow-ийн augmented зургуудыг хасахгүй')
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    print(f"Dataset root: {root}")
    if not root.exists():
        raise FileNotFoundError(f"{root} хавтас байхгүй!")
    
    dfs = []
    
    # 1) GRAZPEDWRI-DX — /dev/shm эсвэл --graz_root-аас уншина
    graz_root = resolve_graz_root(args.graz_root, root)
    if graz_root.exists():
        dfs.append(build_grazpedwri(graz_root))
    else:
        print(f"\n⚠ GRAZPEDWRI-DX/ хавтас байхгүй: {graz_root}")
    
    # 2) FracAtlas
    fa_root = root / 'FracAtlas'
    if fa_root.exists():
        fa_root = resolve_nested(fa_root, ['dataset.csv', 'images'])
        dfs.append(build_fracatlas(fa_root))
    else:
        print(f"\n⚠ FracAtlas/ хавтас байхгүй: {fa_root}")
    
    # 3) BoneFractureCV
    bcv_root = root / 'BoneFractureCV'
    if bcv_root.exists():
        bcv_root = resolve_nested(bcv_root, ['train', 'data.yaml', 'README.dataset.txt'])
        dfs.append(build_yolo_dataset(
            bcv_root, 'bonefracturecv',
            exclude_augmented=not args.include_roboflow_augmented
        ))
    else:
        print(f"\n⚠ BoneFractureCV/ хавтас байхгүй: {bcv_root}")
    
    # 4) MultiRegionXray
    mr_root = root / 'MultiRegionXray'
    if mr_root.exists():
        mr_root = resolve_nested(mr_root, ['train', 'Bone_Fracture_Binary_Classification'])
        dfs.append(build_multiregion(mr_root))
    else:
        print(f"\n⚠ MultiRegionXray/ хавтас байхгүй: {mr_root}")
    
    if not dfs:
        print("\n❌ Нэг ч dataset боловсруулагдсангүй! data_root зөв эсэхийг шалга.")
        return
    
    # Хоосон DataFrame-уудыг хасах
    dfs = [d for d in dfs if len(d) > 0]
    if not dfs:
        print("\n❌ Бүх dataset хоосон гарлаа!")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"НЭГТГЭЛ:")
    print(f"  Нийт зураг       : {len(df):,}")
    print(f"  Fracture (label=1): {(df['label']==1).sum():,} "
          f"({(df['label']==1).mean()*100:.1f}%)")
    print(f"  Non-fracture     : {(df['label']==0).sum():,}")
    print(f"  Unique group     : {df['group_id'].nunique():,}")
    print(f"\n  Dataset тус бүрээр:")
    for ds, g in df.groupby('dataset'):
        pos = (g['label'] == 1).sum()
        print(f"    {ds:20s}: {len(g):>6,} зураг "
              f"({pos:>5,} pos / {len(g)-pos:>5,} neg)")
    print(f"{'='*70}")
    
    # ========================================================================
    # TEST HOLD-OUT (15%)
    # ========================================================================
    from sklearn.model_selection import GroupShuffleSplit
    
    df['strat_key'] = df['dataset'] + '_' + df['label'].astype(str)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio,
                            random_state=args.seed)
    
    train_idx, test_idx = next(gss.split(df, df['label'], df['group_id']))
    
    df_train = df.iloc[train_idx].drop(columns=['strat_key']).reset_index(drop=True)
    df_test  = df.iloc[test_idx].drop(columns=['strat_key']).reset_index(drop=True)
    
    df_train.to_csv(args.train_csv, index=False)
    df_test.to_csv(args.test_csv, index=False)
    
    print(f"\n{'='*70}")
    print(f"TRAIN/TEST SPLIT (Group-level):")
    print(f"{'='*70}")
    
    for name, d in [('TRAIN (CV дотор)', df_train), ('TEST (hold-out)', df_test)]:
        pos = (d['label'] == 1).sum()
        print(f"\n  {name}: {len(d):,} зураг ({pos:,} pos, {pos/len(d)*100:.1f}%)")
        for ds, g in d.groupby('dataset'):
            p = (g['label'] == 1).sum()
            print(f"    {ds:20s}: {len(g):>6,} ({p:>4,} pos)")
    
    overlap = set(df_train['group_id']) & set(df_test['group_id'])
    print(f"\n  Group overlap (заавал 0): {len(overlap)}")
    
    train_pos = (df_train['label'] == 1).sum()
    train_neg = (df_train['label'] == 0).sum()
    pos_weight = train_neg / max(train_pos, 1)
    print(f"\n  Санал болгох pos_weight (BCE): {pos_weight:.3f}")
    print(f"  → config.py-д CFG.pos_weight = {pos_weight:.3f} болгоно уу")
    
    print(f"\n  ✓ Хадгалсан: {args.train_csv}")
    print(f"  ✓ Хадгалсан: {args.test_csv}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
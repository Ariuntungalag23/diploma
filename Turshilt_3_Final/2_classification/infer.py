"""
================================================================================
INFERENCE — TTA + 3-fold ensemble + threshold-based decision
================================================================================
Хэрэглэх:
    python infer.py                       # data_test.csv дээр ажиллуулах
    python infer.py --test_csv my.csv     # өөр CSV
================================================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)

from config import CFG
from data import FractureDataset, build_tta_transform
from model import FractureModel


# ============================================================================
# TTA INFERENCE
# ============================================================================

@torch.no_grad()
def tta_predict(model, df: pd.DataFrame, cfg: CFG, device) -> np.ndarray:
    """
    Multi-scale + hflip TTA.
    
    Жишээ: scales=(320, 384, 448), hflip=True → 6 view averaging
    """
    model.eval()
    all_view_probs = []
    
    views = []
    for s in cfg.tta_scales:
        views.append((s, False))
        if cfg.tta_hflip:
            views.append((s, True))
    
    for scale, flip in views:
        tf = build_tta_transform(scale, hflip=flip)
        ds = FractureDataset(df, transform=tf, cfg=cfg)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        
        batch_probs = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            with autocast(device_type='cuda', enabled=cfg.fp16):
                logits = model(x)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            batch_probs.append(probs)
        
        view_probs = np.concatenate(batch_probs)
        all_view_probs.append(view_probs)
        print(f"    TTA view: scale={scale} flip={flip} done")
    
    return np.mean(np.stack(all_view_probs, axis=0), axis=0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Test CSV (image_path багана зайлшгүй)')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default=None,
                        help='Гаралтын CSV (default: <save_dir>/test_predictions.csv)')
    args = parser.parse_args()
    
    cfg = CFG()
    if args.test_csv: cfg.test_csv = args.test_csv
    if args.save_dir: cfg.save_dir = args.save_dir
    
    save_dir = Path(cfg.save_dir)
    out_path = Path(args.output) if args.output else save_dir / 'test_predictions.csv'
    
    device = torch.device(cfg.device)
    
    print(f"{'='*70}")
    print(f"INFERENCE — {cfg.n_folds}-fold ensemble + TTA")
    print(f"{'='*70}")
    
    # ---- Test CSV ----
    if not Path(cfg.test_csv).exists():
        raise FileNotFoundError(f"{cfg.test_csv} байхгүй!")
    df_test = pd.read_csv(cfg.test_csv)
    has_labels = 'label' in df_test.columns
    print(f"\nTest data: {len(df_test):,} зураг")
    if has_labels:
        print(f"  Label-тай (үнэлгээ хийгдэнэ): {(df_test['label']==1).sum():,} pos")
    
    # ---- Бүх fold-ийг ачаалж TTA-аар прогнозлох ----
    all_fold_probs = []
    
    for fold in range(cfg.n_folds):
        ckpt_path = save_dir / f'fold{fold}.pth'
        if not ckpt_path.exists():
            print(f"  ⚠ {ckpt_path} байхгүй — алгасаж байна")
            continue
        
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"\nFold {fold+1}: CV AUC = {ckpt['auc']:.4f}, "
              f"best epoch = {ckpt.get('best_epoch', '?')}")
        
        # Model build + state load
        model = FractureModel(cfg)
        try:
            model.load_state_dict(ckpt['state_dict'])
        except RuntimeError as e:
            # torch.compile-аас үүссэн _orig_mod. prefix-ийг хасах
            print(f"  → State dict prefix цэвэрлэж байна...")
            cleaned = {
                k.replace('_orig_mod.', ''): v
                for k, v in ckpt['state_dict'].items()
            }
            model.load_state_dict(cleaned)
        
        model = model.to(device, memory_format=torch.channels_last)
        
        # TTA prediction
        probs = tta_predict(model, df_test, cfg, device)
        all_fold_probs.append(probs)
        
        del model
        torch.cuda.empty_cache()
    
    if not all_fold_probs:
        raise RuntimeError("Нэг ч fold checkpoint олдсонгүй!")
    
    # ---- Ensemble: бүх fold-уудын прогнозыг averaging ----
    ensemble_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)
    
    # ---- Save predictions ----
    out_df = df_test.copy()
    out_df['prob'] = ensemble_probs
    
    # Threshold-ийг OOF-аас унших
    cv_summary_path = save_dir / 'cv_summary.json'
    if cv_summary_path.exists():
        with open(cv_summary_path) as f:
            cv = json.load(f)
        thr = cv['threshold']
        out_df['pred'] = (ensemble_probs >= thr).astype(int)
        print(f"\nThreshold (OOF дээрх калибрелс): {thr:.4f}")
    else:
        thr = 0.5
        out_df['pred'] = (ensemble_probs >= thr).astype(int)
        print(f"\n⚠ cv_summary.json байхгүй → threshold = 0.5 ашиглаж байна")
    
    out_df.to_csv(out_path, index=False)
    print(f"\n  ✓ Прогноз хадгалсан: {out_path}")
    
    # ---- Хэрэв label-тай бол үнэлгээ ----
    if has_labels:
        labels = df_test['label'].values
        
        print(f"\n{'='*70}")
        print(f"ҮНЭЛГЭЭ (TEST SET — сургалтад огт ороогүй)")
        print(f"{'='*70}")
        
        # AUC
        test_auc = roc_auc_score(labels, ensemble_probs)
        print(f"\n  🎯 TEST AUC = {test_auc:.4f}")
        
        # CV vs Test зөрүү (generalization gap)
        if cv_summary_path.exists():
            cv_auc = cv['cv_auc']
            gap = cv_auc - test_auc
            print(f"     CV AUC  = {cv_auc:.4f}")
            print(f"     Gap     = {gap:+.4f}  "
                  f"({'overfit-гүй' if abs(gap) < 0.02 else 'overfit байж магадгүй'})")
        
        # Classification report
        pred = (ensemble_probs >= thr).astype(int)
        print(f"\n  Threshold = {thr:.4f}")
        print(f"\n{classification_report(labels, pred, target_names=['non_fracture', 'fracture'], digits=4)}")
        
        # Confusion matrix
        cm = confusion_matrix(labels, pred)
        print(f"  Confusion Matrix:")
        print(f"                 Pred 0    Pred 1")
        print(f"    True 0    {cm[0,0]:>8d}  {cm[0,1]:>8d}")
        print(f"    True 1    {cm[1,0]:>8d}  {cm[1,1]:>8d}")
        
        # Dataset тус бүрээр AUC (хэрэв dataset багана байгаа бол)
        if 'dataset' in df_test.columns:
            print(f"\n  Dataset тус бүрээр AUC:")
            for ds, g_idx in df_test.groupby('dataset').groups.items():
                idx = list(g_idx)
                if len(set(labels[idx])) > 1:
                    a = roc_auc_score(labels[idx], ensemble_probs[idx])
                    n = len(idx)
                    pos = labels[idx].sum()
                    print(f"    {ds:20s}: AUC {a:.4f}  (n={n:>5d}, pos={int(pos):>4d})")
        
        # Үр дүнг save
        result = {
            'test_auc': float(test_auc),
            'threshold': float(thr),
            'n_test': len(df_test),
            'n_positive': int(labels.sum()),
            'confusion_matrix': cm.tolist(),
        }
        if cv_summary_path.exists():
            result['cv_auc'] = float(cv['cv_auc'])
            result['generalization_gap'] = float(cv['cv_auc'] - test_auc)
        
        with open(save_dir / 'test_summary.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  ✓ Үнэлгээний тайлан: {save_dir / 'test_summary.json'}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

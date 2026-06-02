"""
================================================================================
BONE FRACTURE BINARY CLASSIFICATION — Strong Recipe for AUC ≥ 0.95
================================================================================
Hardware  : NVIDIA A6000 48GB
Dataset   : ~35K X-ray images (GRAZPEDWRI + FracAtlas + 2 YOLO datasets)
Imbalance : fractured : not_fractured = 1 : 4.7
Target    : Test AUC ≥ 0.95 (TTA), Recall ≥ 0.90 + highest precision threshold

================================================================================
KEY IMPROVEMENTS OVER PREVIOUS RUN (ConvNeXt-Small + focal loss)
================================================================================
1. ConvNeXt-BASE (2x larger than Small) — fits in 48GB at 384, batch 32
2. EMA model averaging (decay 0.999) — almost always +0.3-0.7% AUC
3. Layer-wise LR decay (0.75) — backbone early layers learn slower
4. 5-fold StratifiedKFold ensemble — +0.5-1.5% AUC from averaging
5. CRITICAL: Patient/study-level split (NOT random!) — prevents data leakage
6. Stronger augmentation: RandAugment + Mixup + CutMix + CLAHE
7. BCEWithLogitsLoss with pos_weight=4.7 (simpler than focal, often better)
8. Multi-scale TTA: {320, 384, 448} × {orig, hflip} = 6 views

Usage:
    1. python train_fracture.py --csv data.csv --mode build_csv  (one-time)
    2. python train_fracture.py --csv data.csv --mode train     (5-fold CV)
    3. python train_fracture.py --csv data.csv --mode inference (TTA + threshold)
"""

import os
import json
import math
import argparse
import random
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class CFG:
    # ---- Data ----
    csv_path: str = 'data.csv'      # Columns: image_path, label, group_id (optional)
    img_size: int = 384
    num_workers: int = 8

    # ---- Model ----
    model_name: str = 'convnext_base.fb_in22k_ft_in1k_384'
    dropout: float = 0.2
    drop_path_rate: float = 0.2

    # ---- Training ----
    n_folds: int = 5
    epochs: int = 15
    batch_size: int = 32              # ConvNeXt-Base at 384 needs ~24GB
    accumulate: int = 2               # Effective batch 64
    grad_clip: float = 1.0

    # ---- Optimizer (layer-wise decay) ----
    lr_head: float = 5e-4
    lr_backbone: float = 5e-5
    layer_decay: float = 0.75
    weight_decay: float = 0.02
    warmup_epochs: int = 1

    # ---- Loss ----
    pos_weight: float = 4.7           # 1:4.7 imbalance
    label_smoothing: float = 0.05

    # ---- Augmentation ----
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5             # P(apply mixup OR cutmix per batch)

    # ---- EMA ----
    ema_decay: float = 0.999

    # ---- TTA ----
    tta_scales: tuple = (320, 384, 448)
    tta_hflip: bool = True

    # ---- Misc ----
    seed: int = 42
    device: str = 'cuda'
    save_dir: str = 'checkpoints'
    fp16: bool = True
    compile: bool = True              # torch.compile


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# DATASET
# ============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool):
    """Strong augmentation for medical X-ray."""
    if train:
        return A.Compose([
            A.LongestMaxSize(max_size=int(img_size * 1.15)),
            A.PadIfNeeded(min_height=int(img_size * 1.15),
                          min_width=int(img_size * 1.15),
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.15,
                                rotate_limit=20, border_mode=cv2.BORDER_CONSTANT,
                                p=0.7),
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                            contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.3),
            A.CoarseDropout(max_holes=3, max_height=img_size // 10,
                             max_width=img_size // 10, fill_value=0, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


class FractureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths  = df['image_path'].values
        self.labels = df['label'].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, torch.tensor(self.labels[idx])


# ============================================================================
# MIXUP / CUTMIX
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    x_cm = x.clone()
    x_cm[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    real_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_cm = real_lam * y + (1 - real_lam) * y[idx]
    return x_cm, y_cm


# ============================================================================
# EMA MODEL
# ============================================================================

class ModelEMA:
    """Exponential moving average of model weights."""
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1 - d))


# ============================================================================
# MODEL
# ============================================================================

class FractureModel(nn.Module):
    def __init__(self, model_name, dropout=0.2, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,            # remove head
            drop_path_rate=drop_path_rate,
            global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),   # binary logit
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f).squeeze(-1)


# ============================================================================
# OPTIMIZER WITH LAYER-WISE LR DECAY
# ============================================================================

def get_layer_id_convnext(name: str, num_stages: int = 4) -> int:
    """Map parameter name → layer id (smaller = earlier = lower LR)."""
    if name.startswith('head'):
        return num_stages + 1
    if 'stages' in name:
        # e.g. backbone.stages.2.blocks.5.norm.weight
        parts = name.split('.')
        for p_i, p in enumerate(parts):
            if p == 'stages' and p_i + 1 < len(parts):
                try:
                    return int(parts[p_i + 1]) + 1
                except ValueError:
                    pass
    return 0  # stem


def build_optimizer(model, cfg: CFG, num_stages=4):
    """AdamW with layer-wise LR decay (LLD)."""
    decay = cfg.layer_decay
    no_decay = ('bias', 'norm', 'gamma', 'beta')
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = get_layer_id_convnext(name, num_stages)
        # head_layer_id = num_stages + 1 → lr = lr_head
        # backbone layer 0..num_stages → scaled
        if layer_id >= num_stages + 1:
            lr = cfg.lr_head
        else:
            scale = decay ** (num_stages - layer_id)
            lr = cfg.lr_backbone * scale
        wd = 0.0 if any(nd in name for nd in no_decay) else cfg.weight_decay
        params.append({'params': p, 'lr': lr, 'weight_decay': wd, 'layer_id': layer_id})

    optimizer = torch.optim.AdamW(params)
    return optimizer


# ============================================================================
# TRAINING
# ============================================================================

def smooth_label(y, eps):
    return y * (1 - eps) + 0.5 * eps


def train_one_epoch(model, loader, optimizer, scaler, criterion, ema, cfg, device):
    model.train()
    total_loss = 0.0; n = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True).float()
        y = smooth_label(y, cfg.label_smoothing)

        # Mixup or cutmix (mutually exclusive per batch)
        if random.random() < cfg.mix_prob:
            if random.random() < 0.5:
                x, y = mixup_data(x, y, cfg.mixup_alpha)
            else:
                x, y = cutmix_data(x, y, cfg.cutmix_alpha)

        with autocast(enabled=cfg.fp16):
            logits = model(x)
            loss = criterion(logits, y) / cfg.accumulate

        scaler.scale(loss).backward()
        if (step + 1) % cfg.accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * cfg.accumulate; n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, fp16=True):
    model.eval()
    all_logits, all_y = [], []
    total_loss = 0.0; n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True).float()
        with autocast(enabled=fp16):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item(); n += 1
        all_logits.append(logits.float().cpu()); all_y.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_y).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc = roc_auc_score(labels, probs)
    return total_loss / max(n, 1), auc, probs, labels


def train_one_fold(fold, df_train, df_val, cfg: CFG, device):
    """Train a single fold, return EMA model + OOF predictions."""
    print(f'\n{"="*70}\n  FOLD {fold+1}/{cfg.n_folds}\n{"="*70}')

    train_tf = build_transforms(cfg.img_size, train=True)
    val_tf   = build_transforms(cfg.img_size, train=False)
    train_ds = FractureDataset(df_train, transform=train_tf)
    val_ds   = FractureDataset(df_val,   transform=val_tf)

    # Weighted sampler — heavy minority oversampling
    pos = (df_train['label'] == 1).sum()
    neg = (df_train['label'] == 0).sum()
    w_pos = 1.0 / pos; w_neg = 1.0 / neg
    weights = np.where(df_train['label'].values == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(weights, num_samples=len(df_train),
                                      replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               sampler=sampler, num_workers=cfg.num_workers,
                               pin_memory=True, drop_last=True,
                               persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                               shuffle=False, num_workers=cfg.num_workers,
                               pin_memory=True, persistent_workers=True)

    # Model
    model = FractureModel(cfg.model_name, cfg.dropout, cfg.drop_path_rate)
    model = model.to(device, memory_format=torch.channels_last)
    if cfg.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    ema = ModelEMA(model, decay=cfg.ema_decay)

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accumulate)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.fp16)

    # Loss (BCE with pos_weight — simpler than focal, often better)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_val = nn.BCEWithLogitsLoss()  # no pos_weight for val loss

    best_auc = -1.0; best_state = None
    for ep in range(cfg.epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                    criterion, ema, cfg, device)
        scheduler.step()
        # Eval with EMA
        vl_loss, vl_auc, _, _ = validate(ema.module, val_loader,
                                          criterion_val, device, cfg.fp16)
        mark = ''
        if vl_auc > best_auc:
            best_auc = vl_auc
            best_state = {k: v.detach().cpu().clone()
                           for k, v in ema.module.state_dict().items()}
            mark = ' ★'
        print(f'[Fold {fold+1}][Ep {ep+1:>2}/{cfg.epochs}] '
              f'trL {tr_loss:.4f}  vlL {vl_loss:.4f}  AUC {vl_auc:.4f}{mark}')

    # Save best
    save_path = Path(cfg.save_dir) / f'fold{fold}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'auc': best_auc,
                 'cfg': asdict(cfg), 'fold': fold}, save_path)
    print(f'   ✓ Saved {save_path} | best AUC = {best_auc:.4f}')

    # OOF predictions using best state
    ema.module.load_state_dict(best_state)
    _, oof_auc, oof_probs, oof_labels = validate(ema.module, val_loader,
                                                   criterion_val, device, cfg.fp16)
    return ema.module, oof_probs, oof_labels, best_auc


# ============================================================================
# TTA INFERENCE
# ============================================================================

@torch.no_grad()
def tta_predict(model, df, cfg: CFG, device):
    """Multi-scale + hflip TTA — returns averaged probabilities."""
    model.eval()
    all_probs = []
    for scale in cfg.tta_scales:
        for flip in ([False, True] if cfg.tta_hflip else [False]):
            tf = build_transforms(scale, train=False)
            if flip:
                tf = A.Compose(tf.transforms[:-2] + [A.HorizontalFlip(p=1.0)]
                                + list(tf.transforms[-2:]))
            ds = FractureDataset(df, transform=tf)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)
            view_probs = []
            for x, _ in loader:
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
                with autocast(enabled=cfg.fp16):
                    logits = model(x)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                view_probs.append(probs)
            view_probs = np.concatenate(view_probs)
            all_probs.append(view_probs)
            print(f'   TTA scale={scale} flip={flip} done')
    return np.mean(np.stack(all_probs, axis=0), axis=0)


# ============================================================================
# THRESHOLD CALIBRATION
# ============================================================================

def find_threshold(probs, labels, target_recall=0.90):
    """Find highest-precision threshold s.t. recall >= target_recall."""
    prec, rec, thr = precision_recall_curve(labels, probs)
    # prec/rec are length N+1, thr is length N — drop the last entry
    prec, rec = prec[:-1], rec[:-1]
    valid = rec >= target_recall
    if not valid.any():
        # Fall back to best F1
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        idx = int(np.argmax(f1))
        return float(thr[idx]), {'recall': float(rec[idx]),
                                  'precision': float(prec[idx]),
                                  'f1': float(f1[idx])}
    # Highest precision among valid thresholds
    valid_prec = np.where(valid, prec, -1.0)
    idx = int(np.argmax(valid_prec))
    return float(thr[idx]), {'recall': float(rec[idx]),
                              'precision': float(prec[idx]),
                              'f1': float(2 * prec[idx] * rec[idx] /
                                           (prec[idx] + rec[idx] + 1e-9))}


# ============================================================================
# CV ORCHESTRATION
# ============================================================================

def run_cv(cfg: CFG):
    device = torch.device(cfg.device)
    df = pd.read_csv(cfg.csv_path)
    assert {'image_path', 'label'}.issubset(df.columns), \
        "CSV must have 'image_path' and 'label' columns"

    use_groups = 'group_id' in df.columns
    if use_groups:
        print(f'🔑 Using STUDY/PATIENT-LEVEL split (group_id column) — '
              f'prevents data leakage')
        splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True,
                                          random_state=cfg.seed)
        split_iter = splitter.split(df, df['label'], df['group_id'])
    else:
        print(f'⚠ WARNING: No group_id column — using random Stratified split.')
        print(f'  If multiple images per patient/study exist → LEAKAGE.')
        splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True,
                                     random_state=cfg.seed)
        split_iter = splitter.split(df, df['label'])

    oof_probs = np.zeros(len(df))
    oof_labels = df['label'].values
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(split_iter):
        df_train = df.iloc[tr_idx].reset_index(drop=True)
        df_val   = df.iloc[val_idx].reset_index(drop=True)
        print(f'\nFold {fold+1}: train={len(df_train)} '
               f'(pos {(df_train["label"]==1).sum()}), '
               f'val={len(df_val)} '
               f'(pos {(df_val["label"]==1).sum()})')

        _, probs, labels, best_auc = train_one_fold(fold, df_train, df_val,
                                                     cfg, device)
        oof_probs[val_idx] = probs
        fold_aucs.append(best_auc)
        # Free memory between folds
        torch.cuda.empty_cache()

    # OOF AUC
    cv_auc = roc_auc_score(oof_labels, oof_probs)
    print(f'\n{"="*70}')
    print(f'  5-FOLD CV OOF AUC = {cv_auc:.4f}  '
          f'(mean per-fold {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f})')
    print(f'{"="*70}')

    # Threshold calibration on OOF
    thr, m = find_threshold(oof_probs, oof_labels, target_recall=0.90)
    print(f'  Threshold @ recall>=0.90 : {thr:.4f}')
    print(f'    recall    = {m["recall"]:.4f}')
    print(f'    precision = {m["precision"]:.4f}')
    print(f'    f1        = {m["f1"]:.4f}')

    # Save OOF
    out_csv = Path(cfg.save_dir) / 'oof.csv'
    pd.DataFrame({'image_path': df['image_path'],
                  'label': oof_labels,
                  'prob': oof_probs}).to_csv(out_csv, index=False)
    print(f'  ✓ Saved {out_csv}')

    with open(Path(cfg.save_dir) / 'cv_summary.json', 'w') as f:
        json.dump({'cv_auc': float(cv_auc),
                   'fold_aucs': [float(a) for a in fold_aucs],
                   'threshold': thr,
                   'threshold_metrics': m,
                   'cfg': asdict(cfg)}, f, indent=2)

    return cv_auc


# ============================================================================
# INFERENCE (load all folds, ensemble + TTA)
# ============================================================================

def run_inference(cfg: CFG, test_csv: str):
    """Load 5-fold ensemble, run TTA on test set."""
    device = torch.device(cfg.device)
    df_test = pd.read_csv(test_csv)

    all_fold_probs = []
    for fold in range(cfg.n_folds):
        ckpt_path = Path(cfg.save_dir) / f'fold{fold}.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f'Loading fold {fold} (CV AUC {ckpt["auc"]:.4f}) ...')
        model = FractureModel(cfg.model_name, cfg.dropout, cfg.drop_path_rate)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device, memory_format=torch.channels_last)
        probs = tta_predict(model, df_test, cfg, device)
        all_fold_probs.append(probs)
        del model; torch.cuda.empty_cache()

    ensemble_probs = np.mean(np.stack(all_fold_probs, axis=0), axis=0)

    out_path = Path(cfg.save_dir) / 'test_predictions.csv'
    df_test = df_test.copy()
    df_test['prob'] = ensemble_probs
    df_test.to_csv(out_path, index=False)
    print(f'\n✓ Test predictions saved: {out_path}')

    # If test has labels → evaluate
    if 'label' in df_test.columns:
        auc = roc_auc_score(df_test['label'], ensemble_probs)
        print(f'\n  TEST AUC = {auc:.4f}')
        # Use OOF-calibrated threshold
        cv_summary = json.load(open(Path(cfg.save_dir) / 'cv_summary.json'))
        thr = cv_summary['threshold']
        pred = (ensemble_probs >= thr).astype(int)
        from sklearn.metrics import classification_report
        print(f'\n  Threshold (from OOF): {thr:.4f}')
        print(classification_report(df_test['label'], pred,
                                     target_names=['not_fractured', 'fractured']))


# ============================================================================
# CSV BUILDER (one-time)
# ============================================================================

def build_csv():
    """Helper to build data.csv from 4 datasets.

    Combines into single CSV with columns:
      image_path, label, group_id (patient/study ID — CRITICAL!)

    USAGE: Edit DATASET_ROOTS and per-dataset logic to match your structure,
    then run this once to generate data.csv.
    """
    DATASET_ROOTS = {
        'grazpedwri': '/path/to/GRAZPEDWRI-DX',
        'fracatlas':  '/path/to/FracAtlas',
        'bonefracturecv': '/path/to/BoneFractureCVProject',
        'yolobonefracture': '/path/to/YOLOBoneFracture',
    }

    rows = []

    # --- GRAZPEDWRI: study-level group_id (CRITICAL — multiple views/patient) ---
    # CSV-д: patient_id column байгаа → group_id болгох
    # df_g = pd.read_csv(f"{DATASET_ROOTS['grazpedwri']}/dataset.csv")
    # for _, r in df_g.iterrows():
    #     rows.append({
    #         'image_path': f"{DATASET_ROOTS['grazpedwri']}/images/{r['filestem']}.png",
    #         'label': int(r['fracture'] > 0),
    #         'group_id': f"graz_{r['patient_id']}",
    #     })

    # --- FracAtlas: file-level (each image is independent) ---
    # for split in ['train', 'val', 'test']:
    #     img_dir = Path(DATASET_ROOTS['fracatlas']) / split / 'img'
    #     ann_dir = Path(DATASET_ROOTS['fracatlas']) / split / 'ann'
    #     for ann in ann_dir.glob('*.json'):
    #         img_name = ann.stem
    #         d = json.load(open(ann))
    #         label = int(any(o['classTitle'] == 'fractured' for o in d.get('objects', [])))
    #         rows.append({
    #             'image_path': str(img_dir / img_name),
    #             'label': label,
    #             'group_id': f"frac_{img_name}",  # no patient grouping available
    #         })

    # --- YOLO datasets: any non-empty .txt → fracture=1 ---
    # for ds_key in ['bonefracturecv', 'yolobonefracture']:
    #     root = Path(DATASET_ROOTS[ds_key])
    #     for img_path in root.rglob('*.jpg'):
    #         label_path = img_path.with_suffix('.txt')
    #         if label_path.parent.name == 'images':
    #             label_path = label_path.parent.parent / 'labels' / label_path.name
    #         label = 0
    #         if label_path.exists() and label_path.stat().st_size > 0:
    #             label = 1
    #         rows.append({
    #             'image_path': str(img_path),
    #             'label': label,
    #             'group_id': f"{ds_key}_{img_path.stem}",
    #         })

    df = pd.DataFrame(rows)
    df.to_csv('data.csv', index=False)
    print(f'Saved data.csv: {len(df)} rows, '
          f'{df["label"].sum()} positive ({df["label"].mean()*100:.1f}%)')
    print(f'Unique groups: {df["group_id"].nunique()}')


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build_csv', 'train', 'inference'],
                         required=True)
    parser.add_argument('--csv', default='data.csv')
    parser.add_argument('--test_csv', default=None,
                         help='Test CSV for inference mode')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    cfg = CFG(csv_path=args.csv, save_dir=args.save_dir,
              epochs=args.epochs, n_folds=args.n_folds)
    seed_all(cfg.seed)

    print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)} '
          f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')

    if args.mode == 'build_csv':
        build_csv()
    elif args.mode == 'train':
        run_cv(cfg)
    elif args.mode == 'inference':
        if args.test_csv is None:
            print('Inference mode requires --test_csv')
            return
        run_inference(cfg, args.test_csv)


if __name__ == '__main__':
    main()

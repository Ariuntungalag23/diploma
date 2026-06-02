"""
================================================================================
TRAIN — 3-fold StratifiedGroupKFold cross-validation
================================================================================
Хэрэглэх:
    python train.py                        # config.py-ийн default ашиглана
    python train.py --epochs 15            # epoch өөрчлөх
    python train.py --no-wandb             # WandB-гүй ажиллуулах
================================================================================
"""

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve

from config import CFG
from data import FractureDataset, build_transforms
from model import FractureModel, ModelEMA, build_optimizer

# WandB optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # хурдан хувилбар
    torch.backends.cudnn.benchmark = True


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
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x_cm = x.clone()
    x_cm[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
    real_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_cm = real_lam * y + (1 - real_lam) * y[idx]
    return x_cm, y_cm


# ============================================================================
# LABEL SMOOTHING
# ============================================================================

def smooth_label(y, eps=0.05):
    """y ∈ {0, 1} → y * (1-eps) + 0.5 * eps"""
    return y * (1 - eps) + 0.5 * eps


# ============================================================================
# THRESHOLD CALIBRATION
# ============================================================================

def find_threshold(probs, labels, target_recall=0.95):
    """
    Recall ≥ target_recall-г хангах хамгийн өндөр precision-тай threshold олох.
    """
    prec, rec, thr = precision_recall_curve(labels, probs)
    # prec, rec — урт N+1, thr — урт N → сүүлийн утгыг хасах
    prec, rec = prec[:-1], rec[:-1]
    
    valid = rec >= target_recall
    if not valid.any():
        # Recall thresholding боломжгүй бол F1-ийн хамгийн дээдийг сонгох
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        idx = int(np.argmax(f1))
        return float(thr[idx]), {
            'recall': float(rec[idx]),
            'precision': float(prec[idx]),
            'f1': float(f1[idx]),
            'note': f'recall ≥ {target_recall} боломжгүй, F1 хамгийн их-ийг сонгосон',
        }
    
    # Recall шалгуурыг хангадаг threshold-уудаас precision хамгийн дээд
    valid_prec = np.where(valid, prec, -1.0)
    idx = int(np.argmax(valid_prec))
    return float(thr[idx]), {
        'recall': float(rec[idx]),
        'precision': float(prec[idx]),
        'f1': float(2 * prec[idx] * rec[idx] / (prec[idx] + rec[idx] + 1e-9)),
    }


# ============================================================================
# СУРГАЛТ — НЭГ EPOCH
# ============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion, ema, cfg, device, epoch_idx):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True).float()
        y = smooth_label(y, cfg.label_smoothing)
        
        # Mixup / CutMix (тус бүр 50%-ийн магадлалтай)
        if random.random() < cfg.mix_prob:
            if random.random() < 0.5:
                x, y = mixup_data(x, y, cfg.mixup_alpha)
            else:
                x, y = cutmix_data(x, y, cfg.cutmix_alpha)
        
        # Forward + loss
        with autocast(device_type='cuda', enabled=cfg.fp16):
            logits = model(x)
            loss = criterion(logits, y) / cfg.accumulate
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % cfg.accumulate == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)
        
        total_loss += loss.item() * cfg.accumulate
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(model, loader, criterion, device, fp16=True):
    model.eval()
    all_logits, all_y = [], []
    total_loss = 0.0
    n = 0
    
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True).float()
        with autocast(device_type='cuda', enabled=fp16):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item()
        n += 1
        all_logits.append(logits.float().cpu())
        all_y.append(y.cpu())
    
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_y).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    
    auc = roc_auc_score(labels, probs)
    return total_loss / max(n, 1), auc, probs, labels


# ============================================================================
# FOLD СУРГАЛТ
# ============================================================================

def train_one_fold(fold: int, df_train: pd.DataFrame, df_val: pd.DataFrame,
                   cfg: CFG, device, use_wandb: bool):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold+1}/{cfg.n_folds}")
    print(f"{'='*70}")
    print(f"  Train: {len(df_train):,} ({(df_train['label']==1).sum():,} pos)")
    print(f"  Val  : {len(df_val):,} ({(df_val['label']==1).sum():,} pos)")
    
    # Pos_weight-г одоогийн fold-ийн train data-аас бодит тооцох
    train_pos = (df_train['label'] == 1).sum()
    train_neg = (df_train['label'] == 0).sum()
    pos_weight_value = train_neg / max(train_pos, 1)
    print(f"  Pos_weight: {pos_weight_value:.3f}")
    
    # ---- WandB ----
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"fold-{fold+1}",
            config=cfg.to_dict(),
            reinit=True,
        )
    
    # ---- Dataset & Loader ----
    train_tf = build_transforms(cfg.img_size, train=True)
    val_tf   = build_transforms(cfg.img_size, train=False)
    train_ds = FractureDataset(df_train, transform=train_tf, cfg=cfg)
    val_ds   = FractureDataset(df_val,   transform=val_tf,   cfg=cfg)
    
    # Weighted sampler — minority class oversampling
    w_pos = 1.0 / train_pos
    w_neg = 1.0 / train_neg
    weights = np.where(df_train['label'].values == 1, w_pos, w_neg)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(df_train), replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    
    # ---- Model ----
    model = FractureModel(cfg).to(device, memory_format=torch.channels_last)
    if cfg.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
    
    ema = ModelEMA(model, decay=cfg.ema_decay)
    
    # ---- Optimizer + Scheduler ----
    optimizer = build_optimizer(model, cfg)
    
    # Cosine LR with warmup
    steps_per_epoch = math.ceil(len(train_loader) / cfg.accumulate)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda', enabled=cfg.fp16)
    
    # ---- Loss ----
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_val = nn.BCEWithLogitsLoss()
    
    # ---- Training loop ----
    best_auc = -1.0
    best_state = None
    best_epoch = -1
    
    for ep in range(cfg.epochs):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, ema, cfg, device, ep
        )
        
        # EMA model-оор validate
        vl_loss, vl_auc, vl_probs, vl_labels = validate(
            ema.module, val_loader, criterion_val, device, cfg.fp16
        )
        
        # Recall @ target threshold-ыг тооцох (мониторингд)
        thr, m = find_threshold(vl_probs, vl_labels, cfg.target_recall)
        
        mark = ''
        if vl_auc > best_auc:
            best_auc = vl_auc
            best_epoch = ep + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in ema.module.state_dict().items()
            }
            mark = ' ★'
        
        lr_now = optimizer.param_groups[-1]['lr']  # head LR
        print(
            f"  [Fold {fold+1}][Ep {ep+1:>2}/{cfg.epochs}] "
            f"trL {tr_loss:.4f}  vlL {vl_loss:.4f}  AUC {vl_auc:.4f}  "
            f"R@0.95P {m['precision']:.3f}  lr {lr_now:.2e}{mark}"
        )
        
        if use_wandb:
            wandb.log({
                'fold': fold + 1,
                'epoch': ep + 1,
                'train_loss': tr_loss,
                'val_loss': vl_loss,
                'val_auc': vl_auc,
                'val_precision_at_target_recall': m['precision'],
                'val_recall': m['recall'],
                'val_f1': m['f1'],
                'lr_head': lr_now,
            })
    
    # ---- Save best checkpoint ----
    save_path = Path(cfg.save_dir) / f'fold{fold}.pth'
    torch.save({
        'state_dict': best_state,
        'auc': best_auc,
        'best_epoch': best_epoch,
        'fold': fold,
        'cfg': cfg.to_dict(),
    }, save_path)
    print(f"  ✓ Хадгалсан: {save_path}")
    print(f"    Best AUC = {best_auc:.4f} (epoch {best_epoch})")
    
    # ---- OOF prediction ----
    ema.module.load_state_dict(best_state)
    _, oof_auc, oof_probs, oof_labels = validate(
        ema.module, val_loader, criterion_val, device, cfg.fp16
    )
    
    if use_wandb:
        wandb.log({'final_val_auc': oof_auc})
        wandb.finish()
    
    return oof_probs, oof_labels, best_auc


# ============================================================================
# MAIN — 3-FOLD CV
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--n_folds', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    
    # Config (CLI өөрчилбөл шинэчлэх)
    cfg = CFG()
    if args.train_csv:  cfg.train_csv = args.train_csv
    if args.save_dir:   cfg.save_dir = args.save_dir
    if args.epochs:     cfg.epochs = args.epochs
    if args.n_folds:    cfg.n_folds = args.n_folds
    if args.batch_size: cfg.batch_size = args.batch_size
    cfg.use_wandb = cfg.use_wandb and not args.no_wandb
    
    # WandB шалгалт
    use_wandb = cfg.use_wandb
    if use_wandb and not WANDB_AVAILABLE:
        print("⚠ wandb суулгаагүй. pip install wandb. WandB-гүй үргэлжлүүлнэ.")
        use_wandb = False
    
    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    
    print(f"\n{'='*70}")
    print(f"FRACTURE CLASSIFICATION — 3-FOLD CV")
    print(f"{'='*70}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    
    # ---- Data unshix ----
    if not Path(cfg.train_csv).exists():
        raise FileNotFoundError(
            f"{cfg.train_csv} байхгүй! Эхлээд build_dataset.py-г ажиллуул."
        )
    df = pd.read_csv(cfg.train_csv)
    print(f"\nTraining data: {len(df):,} зураг ({(df['label']==1).sum():,} pos)")
    print(f"  Unique groups: {df['group_id'].nunique():,}")
    
    # ---- StratifiedGroupKFold ----
    print(f"\n🔑 Patient-level split (StratifiedGroupKFold) — leakage урьдчилан сэргийлнэ")
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )
    splits = list(sgkf.split(df, df['label'], df['group_id']))
    
    # ---- CV loop ----
    oof_probs = np.zeros(len(df))
    oof_labels = df['label'].values
    fold_aucs = []
    
    for fold, (tr_idx, val_idx) in enumerate(splits):
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_vl = df.iloc[val_idx].reset_index(drop=True)
        
        probs, labels, best_auc = train_one_fold(
            fold, df_tr, df_vl, cfg, device, use_wandb
        )
        
        oof_probs[val_idx] = probs
        fold_aucs.append(best_auc)
        
        torch.cuda.empty_cache()
    
    # ---- OOF результат ----
    cv_auc = roc_auc_score(oof_labels, oof_probs)
    print(f"\n{'='*70}")
    print(f"  {cfg.n_folds}-FOLD CV OOF AUC = {cv_auc:.4f}")
    print(f"    Per-fold AUC: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"    Mean ± std  : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"{'='*70}")
    
    # ---- OOF дээр threshold calibration ----
    thr, m = find_threshold(oof_probs, oof_labels, cfg.target_recall)
    print(f"\n  Threshold @ recall ≥ {cfg.target_recall}: {thr:.4f}")
    print(f"    recall    = {m['recall']:.4f}")
    print(f"    precision = {m['precision']:.4f}")
    print(f"    f1        = {m['f1']:.4f}")
    
    # ---- Хадгалах ----
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({
        'image_path': df['image_path'],
        'label': oof_labels,
        'prob': oof_probs,
    }).to_csv(Path(cfg.save_dir) / 'oof.csv', index=False)
    
    with open(Path(cfg.save_dir) / 'cv_summary.json', 'w') as f:
        json.dump({
            'cv_auc': float(cv_auc),
            'fold_aucs': [float(a) for a in fold_aucs],
            'threshold': thr,
            'threshold_metrics': m,
            'cfg': cfg.to_dict(),
        }, f, indent=2)
    
    print(f"\n  ✓ Дараа дараагийн алхам: python infer.py")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

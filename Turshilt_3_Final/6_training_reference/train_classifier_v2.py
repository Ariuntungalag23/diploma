"""
=============================================================================
FRACTURE CLASSIFIER v2 — A6000 48GB оптимизаци
=============================================================================
Target    : AUC ≥ 0.95
Hardware  : NVIDIA A6000 48GB | 32GB RAM | 4 CPU (Xeon E5-2683 v4)
Dataset   : GRAZPEDWRI-DX + FracAtlas + BoneFractureCVProject + YOLOBoneFracture
Architecture: ConvNeXt-Small (ImageNet-22k pretrained) — 50M params

Хэрэглэх:
  1. kaggle.json тохируул (эсвэл data-г гараар татаж тав)
  2. python train_classifier_v2.py
=============================================================================
"""

import os, sys, json, time, gc, random, math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              precision_recall_curve, f1_score,
                              confusion_matrix, classification_report)
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG — A6000 48GB
# =============================================================================
class Config:
    # ── Зам ──────────────────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent / "classifier_v2"
    DATA_ROOT    = PROJECT_ROOT / "data"
    CKPT_DIR     = PROJECT_ROOT / "checkpoints"

    # Kaggle dataset slug-ууд (татах бол)
    KAGGLE_DATASETS = {
        "grazped":  "jillannahmed/grazpedwri-dx",
        "fracatlas": None,          # Аль хэдийн локалд байна
        "bfcv":     "pkdarabi/bone-fracture-detection-computer-vision-project",
        "yolo_frac": "deepakat002/yolo-object-detection-data-bone-fracture",
    }

    # ── Архитектур ───────────────────────────────────────────────────────────
    ENCODER     = "convnext_small.in22k"   # ImageNet-22K → X-ray transfer сайн
    # Сонголт:
    #   "convnext_small.in22k"            50M  — Зөвлөмж ✓
    #   "efficientnetv2_m.in21k"          54M  — 2-р сонголт
    #   "convnext_base.in22k"             89M  — хүчирхэг, VRAM их

    DROP_RATE   = 0.3
    NUM_CLASSES = 1     # binary → BCEWithLogitsLoss

    # ── A6000 оптимизаци ─────────────────────────────────────────────────────
    IMG_SIZE     = 384   # 224 → 384 (A6000 дээр зохих)
    BATCH_SIZE   = 128   # A6000 48GB → их batch
    NUM_WORKERS  = 4     # 4 CPU
    PIN_MEMORY   = True
    USE_AMP      = True
    COMPILE      = True  # torch.compile() — PyTorch 2.0 speedup ~20%
    CHANNELS_LAST = True # NHWC format — modern GPU-д хурдан

    # ── Сургалт ──────────────────────────────────────────────────────────────
    # 2-шат: эхлээд head, дараа бүгдийг
    STAGE1_EPOCHS = 5    # Encoder хөлдөөж, head л сурна
    STAGE2_EPOCHS = 30   # Бүгдийг сурна
    EPOCHS        = STAGE1_EPOCHS + STAGE2_EPOCHS

    LR_HEAD_S1   = 1e-3         # Stage 1: head
    LR_ENC_S2    = 5e-6         # Stage 2: encoder (маш бага)
    LR_HEAD_S2   = 1e-4         # Stage 2: head
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2

    LABEL_SMOOTH = 0.05  # Label smoothing
    MIXUP_ALPHA  = 0.2   # Mixup augmentation
    FOCAL_GAMMA  = 2.0   # Focal loss γ

    GRAD_CLIP    = 1.0
    SEED         = 42

    # ── Threshold тааруулалт ─────────────────────────────────────────────────
    # Recall ≥ TARGET_RECALL байх нөхцөлд хамгийн өндөр precision threshold
    TARGET_RECALL = 0.90


cfg = Config()
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark   = True
    torch.backends.cudnn.deterministic = False
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

print(f"IMG  : {cfg.IMG_SIZE}x{cfg.IMG_SIZE}")
print(f"Batch: {cfg.BATCH_SIZE}")
print(f"Model: {cfg.ENCODER}")


# =============================================================================
# DATA DOWNLOAD (Kaggle)
# =============================================================================
def download_kaggle(slug, dest):
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        print(f"  [SKIP] {slug} — аль хэдийн татсан")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  Татаж байна: {slug}")
    ret = os.system(f"kaggle datasets download {slug} --path {dest} --unzip -q")
    n = sum(len(fs) for _, _, fs in os.walk(dest))
    print(f"  → {n} файл")


def prepare_data():
    """Dataset татах + CSV үүсгэх"""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)

    rows = []

    # ── 1. FracAtlas (локалд байна) ───────────────────────────────────────
    fa_local = Path(__file__).resolve().parents[3] / "fracatlas-DatasetNinja"
    if fa_local.exists():
        print(f"\n[FracAtlas] {fa_local}")
        for split in ["train", "val", "test"]:
            img_dir = fa_local / split / "img"
            for p in img_dir.glob("*.jpg"):
                rows.append({"path": str(p), "label": 1, "source": "fracatlas"})
        nf_dir = fa_local / "not fractured" / "img"
        for p in nf_dir.glob("*.jpg"):
            rows.append({"path": str(p), "label": 0, "source": "fracatlas_neg"})
        print(f"  → {sum(1 for r in rows if r['source']=='fracatlas')} fractured")
        print(f"  → {sum(1 for r in rows if r['source']=='fracatlas_neg')} not fractured")

    # ── 2. GRAZPEDWRI-DX ─────────────────────────────────────────────────
    graz_dir = cfg.DATA_ROOT / "grazped"
    download_kaggle(cfg.KAGGLE_DATASETS["grazped"], graz_dir)
    _parse_grazped(graz_dir, rows)

    # ── 3. BoneFractureCVProject ──────────────────────────────────────────
    bfcv_dir = cfg.DATA_ROOT / "bfcv"
    download_kaggle(cfg.KAGGLE_DATASETS["bfcv"], bfcv_dir)
    _parse_yolo_folder(bfcv_dir, rows, source="bfcv")

    # ── 4. YOLOBoneFracture ───────────────────────────────────────────────
    yolo_dir = cfg.DATA_ROOT / "yolo_frac"
    download_kaggle(cfg.KAGGLE_DATASETS["yolo_frac"], yolo_dir)
    _parse_yolo_folder(yolo_dir / "data" / "training", rows, source="yolo")

    df = pd.DataFrame(rows).sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
    cnt = Counter(df["label"])
    print(f"\nНийт: {len(df)}  |  fractured={cnt[1]}  not_frac={cnt[0]}")
    print(f"Харьцаа 1:{cnt[0]/max(cnt[1],1):.1f}")

    # Train/val/test split (80/10/10) — FracAtlas test-ийг тусад нь хадгал
    fa_test = df[df["source"] == "fracatlas"].sample(frac=0.1, random_state=cfg.SEED)
    rest    = df.drop(fa_test.index)
    n_val   = int(0.1 * len(rest))
    val_df  = rest.sample(n=n_val, random_state=cfg.SEED)
    train_df = rest.drop(val_df.index)

    train_df.to_csv(cfg.PROJECT_ROOT / "train.csv", index=False)
    val_df.to_csv(cfg.PROJECT_ROOT / "val.csv",   index=False)
    fa_test.to_csv(cfg.PROJECT_ROOT / "test.csv", index=False)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(fa_test)}")
    return train_df, val_df, fa_test


def _parse_grazped(root, rows):
    root = Path(root)
    csv_files = list(root.rglob("*.csv"))
    if not csv_files:
        return
    df = pd.read_csv(csv_files[0])
    img_col  = next((c for c in df.columns if "file" in c.lower()), None)
    lab_col  = next((c for c in df.columns if "fracture" in c.lower() or "label" in c.lower()), None)
    if img_col is None or lab_col is None:
        return
    img_dir = next((p for p in root.rglob("images") if p.is_dir()), root)
    n = 0
    for _, row in df.iterrows():
        path = img_dir / str(row[img_col])
        if path.exists():
            label = 1 if str(row[lab_col]).strip().lower() in ("1", "yes", "fracture", "true") else 0
            rows.append({"path": str(path), "label": label, "source": "grazped"})
            n += 1
    print(f"  [GRAZPED] {n} зураг")


def _parse_yolo_folder(root, rows, source):
    root = Path(root)
    img_dirs = list(root.rglob("images"))
    if not img_dirs:
        img_dirs = [root]
    n_pos = n_neg = 0
    for img_dir in img_dirs:
        for ip in img_dir.rglob("*"):
            if ip.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lp = ip.with_suffix(".txt")
            if not lp.exists():
                lp = ip.parent.parent / "labels" / ip.with_suffix(".txt").name
            if lp.exists() and lp.stat().st_size > 3:
                rows.append({"path": str(ip), "label": 1, "source": source})
                n_pos += 1
            else:
                rows.append({"path": str(ip), "label": 0, "source": source})
                n_neg += 1
    print(f"  [{source}] fractured={n_pos}  neg={n_neg}")


# =============================================================================
# DATASET
# =============================================================================
def get_transforms(split):
    if split == "train":
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=20, border_mode=0, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15,
                               rotate_limit=0, border_mode=0, p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            A.CoarseDropout(max_holes=10, max_height=32,
                            max_width=32, fill_value=0, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class FractureDataset(Dataset):
    def __init__(self, df, split="train"):
        self.df = df.reset_index(drop=True)
        self.tf = get_transforms(split)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        img = self.tf(image=img)["image"]
        return img, torch.tensor(float(row["label"]), dtype=torch.float32)


def make_loader(df, split, sampler=None):
    ds = FractureDataset(df, split)
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=(split == "train"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )


def make_weighted_sampler(df):
    """Class imbalance → WeightedRandomSampler"""
    labels = df["label"].values
    cnt    = Counter(labels)
    w      = {0: 1.0 / cnt[0], 1: 1.0 / cnt[1]}
    weights = [w[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# =============================================================================
# MODEL
# =============================================================================
class FractureClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            cfg.ENCODER,
            pretrained=True,
            num_classes=0,    # features only
            drop_rate=cfg.DROP_RATE,
        )
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(cfg.DROP_RATE),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(cfg.DROP_RATE * 0.5),
            nn.Linear(256, cfg.NUM_CLASSES),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True


# =============================================================================
# LOSS — Focal BCE + Label Smoothing
# =============================================================================
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, label_smooth=0.05):
        super().__init__()
        self.gamma  = gamma
        self.pw     = pos_weight
        self.smooth = label_smooth

    def forward(self, logits, targets):
        # Label smoothing
        targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        pw = self.pw.to(logits.device) if self.pw is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none")
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# =============================================================================
# MIXUP
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_loss(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch):
    model.train()
    total_loss = 0; n = 0
    pbar = tqdm(loader, desc=f"Ep{epoch+1} train", ascii=True, ncols=90)
    for x, y in pbar:
        if cfg.CHANNELS_LAST:
            x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        else:
            x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE)

        x, y_a, y_b, lam = mixup_data(x, y, cfg.MIXUP_ALPHA)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.USE_AMP):
            logits = model(x).squeeze(1)
            loss   = mixup_loss(loss_fn, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}")

    return total_loss / n


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in tqdm(loader, desc="  val", ascii=True, ncols=90):
        if cfg.CHANNELS_LAST:
            x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        else:
            x = x.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.USE_AMP):
            logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    # Recall ≥ 0.90 байх хамгийн сайн threshold тааруулна
    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, probs)
    mask = rec_arr[:-1] >= cfg.TARGET_RECALL
    if mask.any():
        best_thr = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])])
    else:
        best_thr = 0.5

    preds = (probs >= best_thr).astype(int)
    return {
        "auc":       float(auc),
        "threshold": best_thr,
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(np.mean(prec_arr[:-1][mask])) if mask.any() else 0.0,
        "recall":    float(np.mean(rec_arr[:-1][mask]))  if mask.any() else 0.0,
        "probs":     probs,
        "labels":    labels,
    }


# =============================================================================
# TTA (Test-Time Augmentation)
# =============================================================================
@torch.no_grad()
def predict_tta(model, loader, n_aug=8):
    """N augmented version-ийн дундаж probability"""
    tta_tf = A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    model.eval()
    all_paths  = loader.dataset.df["path"].tolist()
    all_labels = loader.dataset.df["label"].tolist()
    all_probs  = np.zeros(len(all_paths))

    for aug_i in range(n_aug):
        for i, (img_path, lbl) in enumerate(zip(all_paths, all_labels)):
            img = np.array(Image.open(img_path).convert("RGB"))
            x = tta_tf(image=img)["image"].unsqueeze(0)
            if cfg.CHANNELS_LAST:
                x = x.to(DEVICE, memory_format=torch.channels_last)
            else:
                x = x.to(DEVICE)
            with autocast(enabled=cfg.USE_AMP):
                p = torch.sigmoid(model(x).squeeze()).item()
            all_probs[i] += p

    all_probs /= n_aug
    labels = np.array(all_labels)
    auc = roc_auc_score(labels, all_probs)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, all_probs)
    mask = rec_arr[:-1] >= cfg.TARGET_RECALL
    best_thr = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.5
    preds = (all_probs >= best_thr).astype(int)
    return {
        "auc":       float(auc),
        "threshold": best_thr,
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "probs":     all_probs,
        "labels":    labels,
    }


# =============================================================================
# LEARNING RATE SCHEDULE — Warmup + Cosine
# =============================================================================
def get_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("FRACTURE CLASSIFIER v2 — A6000 48GB")
    print("=" * 65)

    # ── Data ──────────────────────────────────────────────────────────────
    if (cfg.PROJECT_ROOT / "train.csv").exists():
        print("\n[Data] CSV файл олдлоо — шинээр татахгүй")
        train_df = pd.read_csv(cfg.PROJECT_ROOT / "train.csv")
        val_df   = pd.read_csv(cfg.PROJECT_ROOT / "val.csv")
        test_df  = pd.read_csv(cfg.PROJECT_ROOT / "test.csv")
        cnt = Counter(train_df["label"])
        print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
        print(f"Train ratio 1:{cnt[0]/max(cnt[1],1):.1f}")
    else:
        train_df, val_df, test_df = prepare_data()

    # Class weight for loss
    cnt = Counter(train_df["label"])
    pos_weight = torch.tensor([cnt[0] / max(cnt[1], 1)], dtype=torch.float32)
    print(f"\npos_weight: {pos_weight.item():.2f}")

    # Loaders
    sampler     = make_weighted_sampler(train_df)
    train_loader = make_loader(train_df, "train", sampler=sampler)
    val_loader   = make_loader(val_df, "val")
    test_loader  = make_loader(test_df, "test")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FractureClassifier().to(DEVICE)
    if cfg.CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    if cfg.COMPILE and hasattr(torch, "compile"):
        print("\ntorch.compile() эхэллээ (1-р epoch удаан байна)...")
        model = torch.compile(model)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nParams: {total:.1f}M")

    loss_fn = FocalBCELoss(
        gamma=cfg.FOCAL_GAMMA,
        pos_weight=pos_weight,
        label_smooth=cfg.LABEL_SMOOTH,
    )
    scaler  = GradScaler(enabled=cfg.USE_AMP)
    history = []
    best_auc = 0.0

    # ── STAGE 1: Encoder хөлдөөж head л сурна ────────────────────────────
    print("\n" + "="*65)
    print(f"STAGE 1 — Head warming ({cfg.STAGE1_EPOCHS} epoch, LR={cfg.LR_HEAD_S1:.0e})")
    print("="*65)
    model.freeze_encoder()
    optimizer1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR_HEAD_S1, weight_decay=cfg.WEIGHT_DECAY)
    scheduler1 = get_scheduler(optimizer1, cfg.STAGE1_EPOCHS, 1)

    for epoch in range(cfg.STAGE1_EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer1,
                                  scaler, loss_fn, epoch)
        vl = validate(model, val_loader)
        scheduler1.step()
        elapsed = time.time() - t0
        print(f"\n  Ep{epoch+1:02d} ({elapsed:.0f}s) | loss={tr_loss:.4f} | "
              f"AUC={vl['auc']:.4f}  F1={vl['f1']:.3f}  "
              f"Recall={vl['recall']:.3f}  thr={vl['threshold']:.3f}")
        history.append({"stage": 1, "epoch": epoch + 1, **{k: v for k, v in vl.items()
                                                            if k not in ("probs", "labels")}})
        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            _save(model, optimizer1, epoch, vl, "best_stage1.pt")

    # ── STAGE 2: Бүгдийг сурна ───────────────────────────────────────────
    print("\n" + "="*65)
    print(f"STAGE 2 — Full finetune ({cfg.STAGE2_EPOCHS} epoch)")
    print("="*65)
    model.unfreeze_encoder()
    optimizer2 = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.LR_ENC_S2},
        {"params": model.head.parameters(),    "lr": cfg.LR_HEAD_S2},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scheduler2 = get_scheduler(optimizer2, cfg.STAGE2_EPOCHS, cfg.WARMUP_EPOCHS)

    for epoch in range(cfg.STAGE2_EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer2,
                                  scaler, loss_fn, cfg.STAGE1_EPOCHS + epoch)
        vl = validate(model, val_loader)
        scheduler2.step()
        elapsed = time.time() - t0

        mark = ""
        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            _save(model, optimizer2, cfg.STAGE1_EPOCHS + epoch, vl, "best_model.pt")
            mark = " ★"

        print(f"\n  Ep{cfg.STAGE1_EPOCHS+epoch+1:02d} ({elapsed:.0f}s) | "
              f"loss={tr_loss:.4f} | AUC={vl['auc']:.4f}  "
              f"F1={vl['f1']:.3f}  Recall={vl['recall']:.3f}{mark}")
        history.append({"stage": 2, "epoch": cfg.STAGE1_EPOCHS + epoch + 1,
                        **{k: v for k, v in vl.items()
                           if k not in ("probs", "labels")}})

        torch.cuda.empty_cache(); gc.collect()

    # ── FINAL EVALUATION (TTA) ────────────────────────────────────────────
    print("\n" + "="*65)
    print("FINAL EVALUATION — TTA (8 augmentation дундаж)")
    print("="*65)

    # Best model ачаалах
    ckpt = torch.load(cfg.CKPT_DIR / "best_model.pt",
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    print("\n[Val  TTA]")
    val_tta  = predict_tta(model, val_loader)
    print(f"  AUC={val_tta['auc']:.4f}  F1={val_tta['f1']:.3f}  "
          f"Acc={val_tta['accuracy']:.3f}  thr={val_tta['threshold']:.3f}")

    print("\n[Test TTA]")
    test_tta = predict_tta(model, test_loader)
    preds = (test_tta["probs"] >= test_tta["threshold"]).astype(int)
    print(f"  AUC={test_tta['auc']:.4f}  F1={test_tta['f1']:.3f}  "
          f"Acc={test_tta['accuracy']:.3f}")
    print("\n  Classification Report:")
    print(classification_report(
        test_tta["labels"], preds,
        target_names=["Not Fractured", "Fractured"], digits=3))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(test_tta["labels"], preds))

    # Хадгалах
    pd.DataFrame(history).to_csv(cfg.PROJECT_ROOT / "history.csv", index=False)
    with open(cfg.PROJECT_ROOT / "final_metrics.json", "w") as f:
        json.dump({
            "val_auc_tta":  test_tta["auc"],
            "test_auc_tta": test_tta["auc"],
            "threshold":    test_tta["threshold"],
            "encoder":      cfg.ENCODER,
            "img_size":     cfg.IMG_SIZE,
            "batch_size":   cfg.BATCH_SIZE,
        }, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  BEST VAL AUC  : {best_auc:.4f}")
    print(f"  TEST AUC (TTA): {test_tta['auc']:.4f}")
    print(f"{'='*65}")


def _save(model, optimizer, epoch, metrics, name):
    # compiled model бол _orig_mod-аас state_dict авна
    state = (model._orig_mod.state_dict()
             if hasattr(model, "_orig_mod")
             else model.state_dict())
    torch.save({
        "model":   state,
        "epoch":   epoch + 1,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("probs", "labels")},
        "config": {
            "encoder":  cfg.ENCODER,
            "img_size": cfg.IMG_SIZE,
        },
    }, cfg.CKPT_DIR / name)
    print(f"     → {name} хадгалагдлаа (AUC={metrics['auc']:.4f})")


if __name__ == "__main__":
    main()

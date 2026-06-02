"""
=============================================================================
TRAIN 1/3: MURA CLASSIFIER (хугарал/эрүүл)
=============================================================================
Зорилго:
  Рентген зургаас хугарал байгаа эсэхийг бинар ангилна.

Архитектур:
  EfficientNet-B3 (ImageNet pretrained) + бүрэн холболтын толгой → 2 анги

Нөөц:
  ~1-1.5 цаг сургалт, ~3 GB VRAM (A100 дээр)
=============================================================================
"""

import os, sys, time, json, gc, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent / "fracture_thesis"
    FILTERED_DIR = PROJECT_ROOT / "filtered_lists"
    CKPT_DIR     = PROJECT_ROOT / "checkpoints"

    IMG_SIZE     = 224
    BATCH_SIZE   = 192       # MURA-д SAM байхгүй учир их batch
    EPOCHS       = 10
    LR           = 3e-4
    LR_ENCODER   = 3e-5
    WEIGHT_DECAY = 1e-4

    USE_AMP      = True
    NUM_WORKERS  = 8
    SEED         = 42

cfg = Config()
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL
# =============================================================================
class MURAClassifier(nn.Module):
    def __init__(self, n_classes=2, drop=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3", pretrained=True,
            features_only=True, out_indices=[4]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(384, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)[0]
        return self.head(feat)


# =============================================================================
# DATASET
# =============================================================================
TRAIN_TF = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.4),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

VAL_TF = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


class MURADataset(Dataset):
    def __init__(self, csv_path, split="train"):
        self.df = pd.read_csv(csv_path)
        self.tf = TRAIN_TF if split == "train" else VAL_TF

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        aug = self.tf(image=img)
        return {
            "image": aug["image"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch):
    model.train()
    total_loss, n = 0, 0
    pbar = tqdm(loader, desc=f"Ep{epoch+1}/{cfg.EPOCHS} train", ascii=True)
    for batch in pbar:
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE)

        with autocast(enabled=cfg.USE_AMP):
            logits = model(x)
            loss = loss_fn(logits, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.3f}")
    return total_loss / n


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for batch in tqdm(loader, desc=f"Ep{epoch+1} val", ascii=True):
        x = batch["image"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        with autocast(enabled=cfg.USE_AMP):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "labels":    labels.tolist(),
        "preds":     preds.tolist(),
        "probs":     probs.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("TRAIN 1/3: MURA CLASSIFIER")
    print("="*70)
    print(f"  Device: {DEVICE}")
    print(f"  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_csv = cfg.FILTERED_DIR / "mura_train_final.csv"
    valid_csv = cfg.FILTERED_DIR / "mura_valid_final.csv"

    if not train_csv.exists():
        print(f"\n[ERR] {train_csv} байхгүй"); sys.exit(1)
    if not valid_csv.exists():
        print(f"\n[WARN] {valid_csv} байхгүй — train-аас 80/20 хуваана")
        df = pd.read_csv(train_csv).sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
        n_tr = int(0.8 * len(df))
        train_csv_split = cfg.PROJECT_ROOT / "mura_train_split.csv"
        valid_csv_split = cfg.PROJECT_ROOT / "mura_valid_split.csv"
        df.iloc[:n_tr].to_csv(train_csv_split, index=False)
        df.iloc[n_tr:].to_csv(valid_csv_split, index=False)
        train_csv, valid_csv = train_csv_split, valid_csv_split

    train_ds = MURADataset(train_csv, "train")
    val_ds   = MURADataset(valid_csv, "val")
    print(f"\n  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    # Model + optim
    model = MURAClassifier().to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.LR_ENCODER},
        {"params": model.head.parameters(),    "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.USE_AMP)

    # Train loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    best_auc = 0; history = []
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, epoch)
        vl = validate(model, val_loader, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n  Ep{epoch+1:02d} ({elapsed:.0f}s) | loss={tr_loss:.3f} | "
              f"acc={vl['accuracy']:.3f}  f1={vl['f1']:.3f}  "
              f"auc={vl['auc_roc']:.3f}  recall={vl['recall']:.3f}")

        history.append({"epoch":epoch+1, "tr_loss":tr_loss,
                        **{k:v for k,v in vl.items() if k not in ('labels','preds','probs')}})

        if vl["auc_roc"] > best_auc:
            best_auc = vl["auc_roc"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch+1,
                "metrics": {k:v for k,v in vl.items() if k not in ('labels','preds','probs')},
                "config": {k:v for k,v in vars(Config).items() if not k.startswith('_')},
            }, cfg.CKPT_DIR / "mura_classifier.pt")

            # Final eval results JSON
            with open(cfg.CKPT_DIR / "mura_eval.json", "w") as f:
                json.dump({
                    "metrics": {k:v for k,v in vl.items() if k not in ('labels','preds','probs')},
                    "confusion_matrix": confusion_matrix(vl["labels"], vl["preds"]).tolist(),
                }, f, indent=2)
            print(f"     → mura_classifier.pt (AUC={best_auc:.3f})")

        torch.cuda.empty_cache(); gc.collect()

    pd.DataFrame(history).to_csv(cfg.CKPT_DIR / "mura_history.csv", index=False)
    print(f"\n[DONE] Best AUC: {best_auc:.3f}")
    print(f"  Checkpoint: {cfg.CKPT_DIR / 'mura_classifier.pt'}")


if __name__ == "__main__":
    main()

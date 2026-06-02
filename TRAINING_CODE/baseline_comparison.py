"""
=============================================================================
BASELINE COMPARISON — Бакалаврын ажлын харьцуулалт
=============================================================================
Загварууд (ижил дата, ижил hyperparameter):
  1. ResNet-50          (ImageNet-1K pretrained)
  2. EfficientNet-B3    (ImageNet-1K pretrained)  ← хуучин MURA загвартай ойролцоо
  3. ConvNeXt-Small     (ImageNet-1K pretrained)
  4. ConvNeXt-Small     (ImageNet-22K pretrained) ← МААНЬ (train_classifier_v2.py)

Ажиллуулах:
  # Бүх загварыг сурга:
  python baseline_comparison.py

  # Аль хэдийн сургасан загваруудыг ачаалж зөвхөн дүн гарга:
  python baseline_comparison.py --eval_only

  # Зөвхөн тодорхой загварыг сурга (хурдан турших):
  python baseline_comparison.py --models resnet50 efficientnet_b3

Гаралт:
  classifier_v2/baseline_results/
    comparison_table.csv          ← thesis-д хэрэглэх хүснэгт
    comparison_chart.png          ← thesis-д хэрэглэх зураг
    {model_name}/best_model.pt    ← checkpoint
=============================================================================
"""

import os, sys, json, time, gc, random, math, argparse, warnings
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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve,
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# =============================================================================
# ХАРЬЦУУЛАХ ЗАГВАРУУД
# =============================================================================
BASELINES = [
    {
        "name":     "ResNet-50",
        "encoder":  "resnet50",
        "pretrain": "imagenet",
        "desc":     "Сонгодог CNN baseline (ImageNet-1K)",
    },
    {
        "name":     "EfficientNet-B3",
        "encoder":  "efficientnet_b3",
        "pretrain": "imagenet",
        "desc":     "Хуучин MURA загвартай ойролцоо (ImageNet-1K)",
    },
    {
        "name":     "ConvNeXt-S-1K",
        "encoder":  "convnext_small.in1k",
        "pretrain": "imagenet1k",
        "desc":     "ConvNeXt-Small ImageNet-1K pretrained",
    },
    {
        "name":     "ConvNeXt-S-22K",
        "encoder":  "convnext_small.in22k",
        "pretrain": "imagenet22k",
        "desc":     "МААНЬ — ConvNeXt-Small ImageNet-22K (21,841 класс)",
    },
]


# =============================================================================
# CONFIG — train_classifier_v2.py-тай ЯЦИГ ИЖИ
# =============================================================================
class Config:
    PROJECT_ROOT   = Path(__file__).resolve().parent / "classifier_v2"
    BASELINE_DIR   = PROJECT_ROOT / "baseline_results"

    IMG_SIZE     = 384
    BATCH_SIZE   = 64       # baseline нь 4 загвар → batch бага
    NUM_WORKERS  = 4
    PIN_MEMORY   = True
    USE_AMP      = True
    CHANNELS_LAST = True

    STAGE1_EPOCHS = 5
    STAGE2_EPOCHS = 30

    LR_HEAD_S1   = 1e-3
    LR_ENC_S2    = 5e-6
    LR_HEAD_S2   = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2

    DROP_RATE    = 0.3
    LABEL_SMOOTH = 0.05
    MIXUP_ALPHA  = 0.2
    FOCAL_GAMMA  = 2.0
    GRAD_CLIP    = 1.0
    SEED         = 42
    TARGET_RECALL = 0.90

cfg = Config()
cfg.BASELINE_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
print(f"Device: {DEVICE}")


# =============================================================================
# DATASET — train_classifier_v2.py-ийн CSV ашиглана
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
    labels  = df["label"].values
    cnt     = Counter(labels)
    weights = [1.0/cnt[int(l)] for l in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# =============================================================================
# MODEL — ижил head, encoder л өөр
# =============================================================================
class BaselineClassifier(nn.Module):
    def __init__(self, encoder_name, drop=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, num_classes=0, drop_rate=drop)
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(256, 1),
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
# LOSS + MIXUP
# =============================================================================
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None, label_smooth=0.05):
        super().__init__()
        self.gamma  = gamma
        self.pw     = pos_weight
        self.smooth = label_smooth

    def forward(self, logits, targets):
        t  = targets * (1 - self.smooth) + 0.5 * self.smooth
        pw = self.pw.to(logits.device) if self.pw is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=pw, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(loss_fn, pred, ya, yb, lam):
    return lam * loss_fn(pred, ya) + (1 - lam) * loss_fn(pred, yb)


# =============================================================================
# SCHEDULE
# =============================================================================
def get_scheduler(optimizer, total_ep, warmup_ep):
    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        prog = (ep - warmup_ep) / (total_ep - warmup_ep)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, ep_label):
    model.train()
    total_loss = 0; n = 0
    for x, y in tqdm(loader, desc=f"  {ep_label}", ascii=True, ncols=85, leave=False):
        if cfg.CHANNELS_LAST:
            x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        else:
            x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE)
        x, ya, yb, lam = mixup_data(x, y, cfg.MIXUP_ALPHA)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.USE_AMP):
            loss = mixup_loss(loss_fn, model(x).squeeze(1), ya, yb, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.GRAD_CLIP)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item(); n += 1
    return total_loss / n


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        if cfg.CHANNELS_LAST:
            x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        else:
            x = x.to(DEVICE, non_blocking=True)
        with autocast(enabled=cfg.USE_AMP):
            logits = model(x).squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, probs)
    mask = rec_arr[:-1] >= cfg.TARGET_RECALL
    best_thr = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.5
    preds = (probs >= best_thr).astype(int)
    return {
        "auc":       float(auc),
        "threshold": best_thr,
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "recall":    float(np.sum((preds == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)),
        "precision": float(np.sum((preds == 1) & (labels == 1)) / max(np.sum(preds == 1), 1)),
        "probs":     probs,
        "labels":    labels,
    }


@torch.no_grad()
def evaluate_tta(model, df, n_aug=8):
    """8-fold TTA дүгнэлт — test set-д ашиглана"""
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
    paths  = df["path"].tolist()
    labels = np.array(df["label"].tolist())
    probs  = np.zeros(len(paths))

    for _ in range(n_aug):
        for i, p in enumerate(paths):
            img = np.array(Image.open(p).convert("RGB"))
            x = tta_tf(image=img)["image"].unsqueeze(0)
            if cfg.CHANNELS_LAST:
                x = x.to(DEVICE, memory_format=torch.channels_last)
            else:
                x = x.to(DEVICE)
            with autocast(enabled=cfg.USE_AMP):
                probs[i] += torch.sigmoid(model(x).squeeze()).item()

    probs /= n_aug
    auc = roc_auc_score(labels, probs)
    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, probs)
    mask = rec_arr[:-1] >= cfg.TARGET_RECALL
    best_thr = float(thr_arr[mask][np.argmax(prec_arr[:-1][mask])]) if mask.any() else 0.5
    preds = (probs >= best_thr).astype(int)
    return {
        "auc_tta":       float(auc),
        "threshold":     best_thr,
        "accuracy_tta":  float(accuracy_score(labels, preds)),
        "f1_tta":        float(f1_score(labels, preds, zero_division=0)),
        "recall_tta":    float(np.sum((preds==1)&(labels==1)) / max(np.sum(labels==1),1)),
        "precision_tta": float(np.sum((preds==1)&(labels==1)) / max(np.sum(preds==1),1)),
        "probs":         probs,
        "labels":        labels,
        "preds":         preds,
    }


# =============================================================================
# НЭГ ЗАГВАРЫГ СУРГА + ҮНЭЛ
# =============================================================================
def train_and_eval(baseline_cfg, train_df, val_df, test_df, pos_weight):
    name    = baseline_cfg["name"]
    encoder = baseline_cfg["encoder"]
    out_dir = cfg.BASELINE_DIR / name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {name}  —  {baseline_cfg['desc']}")
    print(f"{'='*65}")

    # ── Checkpoint байвал дахин сургахгүй ─────────────────────────────────
    best_ckpt = out_dir / "best_model.pt"
    if best_ckpt.exists():
        print(f"  [SKIP] Аль хэдийн сургасан: {best_ckpt}")
        model = BaselineClassifier(encoder).to(DEVICE)
        if cfg.CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        val_metrics  = ckpt.get("val_metrics",  {})
        best_val_auc = val_metrics.get("auc", 0.0)
    else:
        # ── Loader ────────────────────────────────────────────────────────
        sampler      = make_weighted_sampler(train_df)
        train_loader = make_loader(train_df, "train", sampler=sampler)
        val_loader   = make_loader(val_df,   "val")

        # ── Model ─────────────────────────────────────────────────────────
        model = BaselineClassifier(encoder, drop=cfg.DROP_RATE).to(DEVICE)
        if cfg.CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {n_params:.1f}M")

        loss_fn = FocalBCELoss(cfg.FOCAL_GAMMA, pos_weight, cfg.LABEL_SMOOTH)
        scaler  = GradScaler(enabled=cfg.USE_AMP)
        best_val_auc = 0.0
        val_metrics  = {}

        # Stage 1
        model.freeze_encoder()
        opt1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.LR_HEAD_S1, weight_decay=cfg.WEIGHT_DECAY)
        sch1 = get_scheduler(opt1, cfg.STAGE1_EPOCHS, 1)

        for ep in range(cfg.STAGE1_EPOCHS):
            t0 = time.time()
            tr_loss = train_one_epoch(model, train_loader, opt1, scaler, loss_fn,
                                      f"S1 Ep{ep+1}")
            vl = validate(model, val_loader)
            sch1.step()
            mark = ""
            if vl["auc"] > best_val_auc:
                best_val_auc = vl["auc"]
                val_metrics  = vl
                _save_ckpt(model, ep, vl, best_ckpt, encoder)
                mark = " ★"
            print(f"  S1 Ep{ep+1:02d} ({time.time()-t0:.0f}s) | loss={tr_loss:.4f} | "
                  f"AUC={vl['auc']:.4f}  F1={vl['f1']:.3f}{mark}")

        # Stage 2
        model.unfreeze_encoder()
        opt2 = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": cfg.LR_ENC_S2},
            {"params": model.head.parameters(),    "lr": cfg.LR_HEAD_S2},
        ], weight_decay=cfg.WEIGHT_DECAY)
        sch2 = get_scheduler(opt2, cfg.STAGE2_EPOCHS, cfg.WARMUP_EPOCHS)

        for ep in range(cfg.STAGE2_EPOCHS):
            t0 = time.time()
            tr_loss = train_one_epoch(model, train_loader, opt2, scaler, loss_fn,
                                      f"S2 Ep{ep+1}")
            vl = validate(model, val_loader)
            sch2.step()
            mark = ""
            if vl["auc"] > best_val_auc:
                best_val_auc = vl["auc"]
                val_metrics  = vl
                _save_ckpt(model, cfg.STAGE1_EPOCHS + ep, vl, best_ckpt, encoder)
                mark = " ★"
            print(f"  S2 Ep{ep+1:02d} ({time.time()-t0:.0f}s) | loss={tr_loss:.4f} | "
                  f"AUC={vl['auc']:.4f}  F1={vl['f1']:.3f}{mark}")
            torch.cuda.empty_cache(); gc.collect()

    # ── Test (TTA) ─────────────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"\n  [Test TTA] ачааллаж байна ...")
    test_m = evaluate_tta(model, test_df, n_aug=8)

    print(f"\n  ── {name} эцсийн дүн ──")
    print(f"  Val  AUC      : {best_val_auc:.4f}")
    print(f"  Test AUC(TTA) : {test_m['auc_tta']:.4f}")
    print(f"  Test F1 (TTA) : {test_m['f1_tta']:.3f}")
    print(f"  Test Recall   : {test_m['recall_tta']:.3f}")
    print(f"  Threshold     : {test_m['threshold']:.3f}")

    # Confusion matrix хадгалах
    cm = confusion_matrix(test_m["labels"], test_m["preds"])
    _save_confusion(cm, name, out_dir)

    # Classification report
    report = classification_report(
        test_m["labels"], test_m["preds"],
        target_names=["Not Fractured", "Fractured"], digits=3)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"{name}\n\n{report}\n\n")
        f.write(f"Val AUC     : {best_val_auc:.4f}\n")
        f.write(f"Test AUC TTA: {test_m['auc_tta']:.4f}\n")

    return {
        "name":          name,
        "encoder":       encoder,
        "pretrain":      baseline_cfg["pretrain"],
        "desc":          baseline_cfg["desc"],
        "params_M":      round(sum(p.numel() for p in model.parameters())/1e6, 1),
        "val_auc":       round(best_val_auc, 4),
        "test_auc_tta":  round(test_m["auc_tta"], 4),
        "test_f1_tta":   round(test_m["f1_tta"], 3),
        "test_recall_tta": round(test_m["recall_tta"], 3),
        "test_prec_tta": round(test_m["precision_tta"], 3),
        "test_acc_tta":  round(test_m["accuracy_tta"], 3),
        "threshold":     round(test_m["threshold"], 3),
    }


def _save_ckpt(model, epoch, metrics, path, encoder_name):
    state = (model._orig_mod.state_dict()
             if hasattr(model, "_orig_mod") else model.state_dict())
    torch.save({
        "model":       state,
        "epoch":       epoch + 1,
        "val_metrics": {k: v for k, v in metrics.items()
                        if k not in ("probs", "labels")},
        "config":      {"encoder": encoder_name, "img_size": cfg.IMG_SIZE},
    }, path)


def _save_confusion(cm, name, out_dir):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Frac", "Frac"])
    ax.set_yticklabels(["Not Frac", "Frac"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# ХАРЬЦУУЛАЛТЫН CHART + TABLE
# =============================================================================
def make_comparison_chart(results: list):
    names    = [r["name"]          for r in results]
    auc      = [r["test_auc_tta"]  for r in results]
    f1       = [r["test_f1_tta"]   for r in results]
    recall   = [r["test_recall_tta"] for r in results]
    prec     = [r["test_prec_tta"] for r in results]

    x    = np.arange(len(names))
    w    = 0.2
    best = int(np.argmax(auc))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Baseline Харьцуулалт — Ясны Хугарал Илрүүлэлт",
                 fontsize=14, color="white", fontweight="bold")

    # ── Bar chart ─────────────────────────────────────────────────────────
    ax = axes[0]
    colors_auc = ["#4FC3F7" if i != best else "#FF7043" for i in range(len(names))]
    b1 = ax.bar(x - 1.5*w, auc,    w, label="AUC (TTA)",    color=colors_auc)
    b2 = ax.bar(x - 0.5*w, f1,     w, label="F1",           color="#81C784", alpha=0.9)
    b3 = ax.bar(x + 0.5*w, recall, w, label="Recall",       color="#FFD54F", alpha=0.9)
    b4 = ax.bar(x + 1.5*w, prec,   w, label="Precision",    color="#CE93D8", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.95, color="#FF5252", ls="--", lw=1.5, label="Target AUC=0.95")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Metric харьцуулалт (Test set, TTA)", color="white", fontsize=11)

    # Value label нэмэх
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7, color="white")

    # ── Params vs AUC scatter ──────────────────────────────────────────────
    ax2 = axes[1]
    params = [r["params_M"] for r in results]
    sc_colors = ["#FF7043" if i == best else "#4FC3F7" for i in range(len(names))]
    for i, (p, a, n) in enumerate(zip(params, auc, names)):
        ax2.scatter(p, a, s=200, c=sc_colors[i], zorder=5)
        ax2.annotate(n, (p, a), textcoords="offset points", xytext=(8, 4),
                     fontsize=8, color="white")
    ax2.axhline(0.95, color="#FF5252", ls="--", lw=1.5, alpha=0.7)
    ax2.set_xlabel("Параметр тоо (сая)", color="white")
    ax2.set_ylabel("Test AUC (TTA)", color="white")
    ax2.set_title("Параметр тоо vs AUC", color="white", fontsize=11)
    ax2.grid(alpha=0.3)

    # Best загвар тэмдэглэх
    patch = mpatches.Patch(color="#FF7043", label=f"Best: {names[best]}")
    ax2.legend(handles=[patch], fontsize=9)

    plt.tight_layout()
    chart_path = cfg.BASELINE_DIR / "comparison_chart.png"
    fig.savefig(chart_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Chart → {chart_path}")


def make_comparison_table(results: list):
    df = pd.DataFrame(results)
    cols = ["name", "encoder", "pretrain", "params_M",
            "val_auc", "test_auc_tta", "test_f1_tta",
            "test_recall_tta", "test_prec_tta", "test_acc_tta", "threshold"]
    df = df[cols]
    df = df.sort_values("test_auc_tta", ascending=False).reset_index(drop=True)

    csv_path = cfg.BASELINE_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    # Console хэвлэлт
    print(f"\n{'='*70}")
    print("  BASELINE ХАРЬЦУУЛАЛТ — ЭЦСИЙН ДҮН")
    print(f"{'='*70}")
    header = f"{'Загвар':<22} {'Params':>7} {'Val AUC':>8} {'AUC TTA':>8} {'F1':>6} {'Recall':>7} {'Prec':>6}"
    print(header)
    print("-" * 70)
    for _, row in df.iterrows():
        mark = " ←BEST" if row.name == 0 else ""
        print(f"{row['name']:<22} {row['params_M']:>6.1f}M "
              f"{row['val_auc']:>8.4f} {row['test_auc_tta']:>8.4f} "
              f"{row['test_f1_tta']:>6.3f} {row['test_recall_tta']:>7.3f} "
              f"{row['test_prec_tta']:>6.3f}{mark}")
    print(f"{'='*70}")
    print(f"  CSV → {csv_path}")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true",
                        help="Checkpoint байвал дахин сургахгүй, зөвхөн дүн гарга")
    parser.add_argument("--models", nargs="+",
                        help="Зөвхөн тодорхой загварыг сурга (encoder нэр)")
    args = parser.parse_args()

    # ── CSV ачаалах ──────────────────────────────────────────────────────────
    csv_dir = cfg.PROJECT_ROOT
    train_csv = csv_dir / "train.csv"
    if not train_csv.exists():
        print(f"\n[ERROR] {train_csv} олдсонгүй.")
        print("  → Эхлээд train_classifier_v2.py ажиллуулж CSV үүсгэнэ үү.")
        sys.exit(1)

    train_df = pd.read_csv(csv_dir / "train.csv")
    val_df   = pd.read_csv(csv_dir / "val.csv")
    test_df  = pd.read_csv(csv_dir / "test.csv")
    cnt = Counter(train_df["label"])
    pos_weight = torch.tensor([cnt[0] / max(cnt[1], 1)], dtype=torch.float32)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"pos_weight: {pos_weight.item():.2f}")

    # ── Аль загварыг сургах ──────────────────────────────────────────────────
    to_run = BASELINES
    if args.models:
        to_run = [b for b in BASELINES if b["encoder"] in args.models]
        print(f"Зөвхөн: {[b['name'] for b in to_run]}")

    if args.eval_only:
        print("\n[eval_only] Checkpoint-аас дүн гарган байна ...")

    # ── Сургах / ачаалах ─────────────────────────────────────────────────────
    import gc
    all_results = []
    for bl in to_run:
        result = train_and_eval(bl, train_df, val_df, test_df, pos_weight)
        all_results.append(result)
        gc.collect(); torch.cuda.empty_cache()

    # ── Харьцуулалт ──────────────────────────────────────────────────────────
    df = make_comparison_table(all_results)
    make_comparison_chart(all_results)

    # JSON хадгалах
    with open(cfg.BASELINE_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    best = df.iloc[0]
    print(f"\nHAMGIIN SAIN: {best['name']}")
    print(f"  Test AUC (TTA): {best['test_auc_tta']:.4f}")
    print(f"  Test F1       : {best['test_f1_tta']:.3f}")
    print(f"  Test Recall   : {best['test_recall_tta']:.3f}")
    if best["test_auc_tta"] >= 0.95:
        print("  ✓ TARGET AUC ≥ 0.95 ХҮРЛЭЭ!")
    else:
        print(f"  ⚡ Target-аас {0.95 - best['test_auc_tta']:.4f} дутуу")

    print(f"\nГаралт: {cfg.BASELINE_DIR}/")


if __name__ == "__main__":
    import gc
    main()

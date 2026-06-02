"""
=============================================================================
TRAIN 2/3: FRACATLAS SEGMENTATION + SEVERITY (SAM-LoRA)
=============================================================================
Зорилго:
  Хугарлын маск + хүндрэлийн зэрэг (Grade 0-3, ordinal regression).

Архитектур:
  EfficientNet-B3 encoder
    └→ SAM-LoRA segmentation head (mask)
        └→ Severity head (encoder feat + mask global → 4-grade ordinal)

Урьдчилсан нөхцөл:
  precompute_sam_embeddings.py ажилласан байх (~6 GB cache)

Нөөц:
  ~1.5 цаг сургалт, ~10-15 GB VRAM (A100 дээр)
=============================================================================
"""

import os, sys, time, json, cv2, math, gc, random, hashlib
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
from segment_anything import sam_model_registry
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                              confusion_matrix)
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    PROJECT_ROOT  = Path(__file__).resolve().parent / "fracture_thesis"
    FILTERED_DIR  = PROJECT_ROOT / "filtered_lists"
    SAM_CKPT      = PROJECT_ROOT / "sam_vit_b.pth"
    SAM_EMB_CACHE = PROJECT_ROOT / "sam_emb_cache"
    CKPT_DIR      = PROJECT_ROOT / "checkpoints"

    IMG_SIZE     = 224
    SAM_SIZE     = 1024

    BATCH_SIZE   = 16
    EPOCHS       = 20
    LR_ENCODER   = 3e-5
    LR_HEADS     = 3e-4
    WEIGHT_DECAY = 1e-4

    LORA_RANK    = 4
    LORA_ALPHA   = 4

    USE_AMP      = True
    USE_SAM_CACHE = True
    NUM_WORKERS  = 8
    SEED         = 42

cfg = Config()
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cache_key(image_path):
    return hashlib.md5(str(image_path).encode()).hexdigest()[:16]


# =============================================================================
# LoRA + SAM SEGMENTATION HEAD
# =============================================================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4, alpha=4):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters(): p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)


def inject_lora(module, rank=4, alpha=4, target_names=("q_proj","v_proj")):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += inject_lora(child, rank, alpha, target_names)
    return n


class SAMSegHead(nn.Module):
    def __init__(self, sam_ckpt, rank=4, alpha=4):
        super().__init__()
        sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
        self.image_encoder  = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder   = sam.mask_decoder

        for p in self.parameters(): p.requires_grad = False

        n_lora = inject_lora(self.mask_decoder, rank=rank, alpha=alpha)
        print(f"  LoRA injected into {n_lora} layers (rank={rank})")

        self.register_buffer("pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1,3,1,1))
        self.register_buffer("pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1,3,1,1))

        self.cache_dir = cfg.SAM_EMB_CACHE
        self.use_cache = cfg.USE_SAM_CACHE

    def _load_cached(self, image_paths):
        embs = []
        for p in image_paths:
            cp = self.cache_dir / f"{cache_key(p)}.npy"
            if cp.exists():
                embs.append(torch.from_numpy(np.load(cp)).float())
            else:
                return None
        return torch.stack(embs).to(self.pixel_mean.device)

    def _encode_live(self, x_norm):
        mean = torch.tensor([0.485,0.456,0.406], device=x_norm.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x_norm.device).view(1,3,1,1)
        raw  = (x_norm * std + mean) * 255.0
        raw  = F.interpolate(raw, size=(cfg.SAM_SIZE, cfg.SAM_SIZE),
                             mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.image_encoder((raw - self.pixel_mean) / self.pixel_std)

    def forward(self, x_norm, bboxes=None, image_paths=None):
        B = x_norm.shape[0]

        image_emb = None
        if self.use_cache and image_paths is not None:
            image_emb = self._load_cached(image_paths)
        if image_emb is None:
            image_emb = self._encode_live(x_norm)

        if bboxes is None:
            cx = cfg.SAM_SIZE // 2; half = cfg.SAM_SIZE // 4
            bboxes_sam = torch.tensor([[cx-half, cx-half, cx+half, cx+half]],
                                      device=x_norm.device).float().repeat(B,1)
        else:
            scale = cfg.SAM_SIZE / cfg.IMG_SIZE
            bboxes_sam = bboxes.float() * scale

        # Per-sample mask decoder (zarim SAM хувилбарт batched bbox crash өгдөг)
        low_res_masks = []
        image_pe = self.prompt_encoder.get_dense_pe()
        for idx in range(B):
            with torch.no_grad():
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=None, boxes=bboxes_sam[idx:idx+1], masks=None)
            low_res, _ = self.mask_decoder(
                image_embeddings=image_emb[idx:idx+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            low_res_masks.append(low_res)
        low_res = torch.cat(low_res_masks, dim=0)
        return F.interpolate(low_res, size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
                             mode="bilinear", align_corners=False)


# =============================================================================
# SEVERITY HEAD
# =============================================================================
class SeverityHead(nn.Module):
    def __init__(self, in_dim=384, n_grades=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim + 1, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_grades - 1),
        )
    def forward(self, feat, mask_logits):
        f = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        m = torch.sigmoid(mask_logits).mean(dim=(1,2,3), keepdim=False).unsqueeze(1)
        return self.fc(torch.cat([f, m], dim=1))


# =============================================================================
# MODEL (FracAtlas only)
# =============================================================================
class FracAtlasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3", pretrained=True,
            features_only=True, out_indices=[4])
        self.seg_head = SAMSegHead(cfg.SAM_CKPT,
                                    rank=cfg.LORA_RANK, alpha=cfg.LORA_ALPHA)
        self.sev_head = SeverityHead(in_dim=384, n_grades=4)

    def forward(self, x, bboxes=None, image_paths=None):
        feat = self.encoder(x)[0]
        mask = self.seg_head(x, bboxes, image_paths=image_paths)
        sev  = self.sev_head(feat, mask)
        return {"feat": feat, "mask": mask, "sev": sev}


# =============================================================================
# LOSSES + helpers
# =============================================================================
def bbox_from_mask(mask):
    B, _, H, W = mask.shape
    boxes = torch.zeros(B, 4, device=mask.device)
    for i in range(B):
        ys, xs = torch.where(mask[i,0] > 0.5)
        if len(xs) > 0:
            boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()],
                                    device=mask.device).float()
        else:
            boxes[i] = torch.tensor([W//4, H//4, 3*W//4, 3*H//4],
                                    device=mask.device).float()
    return boxes


class DiceBCELoss(nn.Module):
    def __init__(self, dice_w=1.0, bce_w=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_w, self.bce_w = dice_w, bce_w
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        smooth = 1.0
        inter = (prob*target).sum(dim=(1,2,3))
        denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = 1 - ((2*inter + smooth) / (denom + smooth)).mean()
        return self.bce_w*bce + self.dice_w*dice


class WeightedOrdinalLoss(nn.Module):
    def __init__(self, n_classes=4, class_weights=None):
        super().__init__()
        self.K = n_classes
        self.register_buffer("w",
            torch.ones(n_classes) if class_weights is None else class_weights)
    def forward(self, logits, target):
        thr = torch.arange(self.K-1, device=target.device).unsqueeze(0)
        bin_target = (target.unsqueeze(1) > thr).float()
        sw = self.w[target]
        loss = F.binary_cross_entropy_with_logits(logits, bin_target, reduction="none")
        return (loss.mean(dim=1) * sw).mean()


def dice_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred*target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).mean().item()


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
], additional_targets={"mask":"mask"})

VAL_TF = A.Compose([
    A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
], additional_targets={"mask":"mask"})


def load_fracatlas_anns(fracatlas_root):
    ann_dir = fracatlas_root / "Annotations" / "COCO JSON"
    if not ann_dir.is_dir():
        ann_dir = fracatlas_root / "Annotations"
    out = {}
    for jf in ann_dir.rglob("*.json"):
        with open(jf) as f: raw = json.load(f)
        if isinstance(raw, dict) and "images" in raw and "annotations" in raw:
            id2name = {im["id"]: im.get("file_name", im.get("filename"))
                       for im in raw["images"]}
            by_img = {}
            for ann in raw["annotations"]:
                segs = ann.get("segmentation", [])
                if not isinstance(segs, list): continue
                for poly_flat in segs:
                    if not isinstance(poly_flat, list) or len(poly_flat) < 6: continue
                    poly = list(zip(poly_flat[0::2], poly_flat[1::2]))
                    by_img.setdefault(ann["image_id"], []).append(poly)
            for img_id, fname in id2name.items():
                if fname: out[fname] = by_img.get(img_id, [])
            continue
        entries = (list(raw["_via_img_metadata"].values())
                   if isinstance(raw, dict) and "_via_img_metadata" in raw
                   else (list(raw.values()) if isinstance(raw, dict) else raw))
        for entry in entries:
            if not isinstance(entry, dict): continue
            fname = entry.get("filename")
            if not fname: continue
            polys = []
            regions = entry.get("regions", {})
            if isinstance(regions, dict): regions = list(regions.values())
            for reg in regions:
                s = reg.get("shape_attributes", {}) if isinstance(reg, dict) else {}
                if s.get("name") == "polygon":
                    xs, ys = s.get("all_points_x", []), s.get("all_points_y", [])
                    if len(xs) >= 3: polys.append(list(zip(xs, ys)))
            out[fname] = polys
    return out


def detect_fracatlas_root(csv_path):
    df = pd.read_csv(csv_path, nrows=1)
    p = Path(df["path"].iloc[0])
    while p.name and p.name not in ("FracAtlas", "images"):
        p = p.parent
    if p.name == "images": p = p.parent
    return p


class FracAtlasDataset(Dataset):
    def __init__(self, csv_path, fracatlas_root, split="train"):
        df = pd.read_csv(csv_path)
        df = df[df["severity"] >= 0].reset_index(drop=True)
        self.df = df
        self.tf = TRAIN_TF if split == "train" else VAL_TF
        self.anns = load_fracatlas_anns(fracatlas_root)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        h, w = img.shape[:2]
        polys = self.anns.get(row["filename"], [])
        mask = np.zeros((h, w), dtype=np.float32)
        for poly in polys:
            cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1.0)
        aug = self.tf(image=img, mask=mask)
        return {
            "image":      aug["image"],
            "mask":       aug["mask"].unsqueeze(0).float(),
            "sev":        torch.tensor(int(row["severity"]), dtype=torch.long),
            "image_path": str(row["path"]),
        }


# =============================================================================
# TRAIN / VAL
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, seg_loss, sev_loss, epoch):
    model.train()
    total = {"loss":0, "seg":0, "sev":0}; n = 0

    pbar = tqdm(loader, desc=f"Ep{epoch+1}/{cfg.EPOCHS} train", ascii=True)
    for batch in pbar:
        x       = batch["image"].to(DEVICE, non_blocking=True)
        y_mask  = batch["mask"].to(DEVICE)
        y_sev   = batch["sev"].to(DEVICE)
        paths   = batch["image_path"]

        with torch.no_grad():
            bboxes = bbox_from_mask(y_mask)

        with autocast(enabled=cfg.USE_AMP):
            out = model(x, bboxes=bboxes, image_paths=paths)
            l_seg = seg_loss(out["mask"], y_mask)
            l_sev = sev_loss(out["sev"],  y_sev)
            loss  = l_seg + 0.5 * l_sev

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer); scaler.update()

        total["loss"] += loss.item()
        total["seg"]  += l_seg.item()
        total["sev"]  += l_sev.item()
        n += 1
        pbar.set_postfix(loss=f"{total['loss']/n:.3f}",
                         seg=f"{total['seg']/n:.3f}",
                         sev=f"{total['sev']/n:.3f}")
    return {k: v/n for k,v in total.items()}


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    dice_sum = 0; dice_n = 0
    sev_corr = sev_adj = sev_n = 0
    all_dice, all_iou = [], []
    sev_true, sev_pred = [], []

    for batch in tqdm(loader, desc=f"Ep{epoch+1} val", ascii=True):
        x      = batch["image"].to(DEVICE)
        y_mask = batch["mask"].to(DEVICE)
        y_sev  = batch["sev"].to(DEVICE)
        paths  = batch["image_path"]
        bboxes = bbox_from_mask(y_mask)

        with autocast(enabled=cfg.USE_AMP):
            out = model(x, bboxes=bboxes, image_paths=paths)

        # Pixel-wise Dice + IoU
        pred = (torch.sigmoid(out["mask"]) > 0.5).float()
        inter = (pred*y_mask).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + y_mask.sum(dim=(1,2,3))
        d = (2*inter + 1e-6) / (union + 1e-6)
        i = (inter + 1e-6) / (union - inter + 1e-6)
        all_dice.extend(d.cpu().numpy())
        all_iou.extend(i.cpu().numpy())
        dice_sum += d.sum().item(); dice_n += y_mask.size(0)

        # Severity
        sp = (torch.sigmoid(out["sev"]) > 0.5).sum(dim=1)
        sev_corr += (sp == y_sev).sum().item()
        sev_adj  += ((sp - y_sev).abs() <= 1).sum().item()
        sev_n    += y_sev.size(0)
        sev_true.extend(y_sev.cpu().numpy())
        sev_pred.extend(sp.cpu().numpy())

    return {
        "mean_dice":   float(np.mean(all_dice)),
        "median_dice": float(np.median(all_dice)),
        "std_dice":    float(np.std(all_dice)),
        "mean_iou":    float(np.mean(all_iou)),
        "sev_exact":   sev_corr / sev_n,
        "sev_adj":     sev_adj  / sev_n,
        "sev_mae":     float(np.abs(np.array(sev_true) - np.array(sev_pred)).mean()),
        "sev_true":    [int(x) for x in sev_true],
        "sev_pred":    [int(x) for x in sev_pred],
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("TRAIN 2/3: FRACATLAS SEG + SEVERITY (SAM-LoRA)")
    print("="*70)
    print(f"  Device: {DEVICE}")
    print(f"  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    frac_csv = cfg.FILTERED_DIR / "fracatlas_final.csv"
    if not frac_csv.exists():
        print(f"\n[ERR] {frac_csv} байхгүй"); sys.exit(1)
    if not cfg.SAM_CKPT.exists():
        print(f"\n[ERR] {cfg.SAM_CKPT} байхгүй"); sys.exit(1)

    fracatlas_root = detect_fracatlas_root(frac_csv)
    print(f"\n  FracAtlas root: {fracatlas_root}")

    df = pd.read_csv(frac_csv).sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
    n_tr = int(0.8 * len(df))
    tr_csv = cfg.PROJECT_ROOT / "_frac_train.csv"
    va_csv = cfg.PROJECT_ROOT / "_frac_val.csv"
    df.iloc[:n_tr].to_csv(tr_csv, index=False)
    df.iloc[n_tr:].to_csv(va_csv, index=False)

    train_ds = FracAtlasDataset(tr_csv, fracatlas_root, "train")
    val_ds   = FracAtlasDataset(va_csv, fracatlas_root, "val")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    model = FracAtlasModel().to(DEVICE)

    if hasattr(model.encoder, "set_grad_checkpointing"):
        model.encoder.set_grad_checkpointing(enable=True)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total : {total/1e6:.1f}M  |  Trainable: {train/1e6:.1f}M")

    sev_weights = torch.tensor([1.0, 2.0, 2.0, 3.0]).to(DEVICE)
    seg_loss = DiceBCELoss(dice_w=1.0, bce_w=1.0)
    sev_loss = WeightedOrdinalLoss(4, sev_weights)

    enc_params  = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = (
        [p for p in model.seg_head.parameters() if p.requires_grad] +
        [p for p in model.sev_head.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": cfg.LR_ENCODER, "weight_decay": 1e-5},
        {"params": head_params, "lr": cfg.LR_HEADS,   "weight_decay": cfg.WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    best_dice = 0; history = []
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, scaler,
                             seg_loss, sev_loss, epoch)
        vl = validate(model, val_loader, epoch)
        scheduler.step(epoch + 1)
        elapsed = time.time() - t0

        print(f"\n  Ep{epoch+1:02d} ({elapsed:.0f}s) | tr_loss={tr['loss']:.3f} | "
              f"dice={vl['mean_dice']:.3f} (med={vl['median_dice']:.3f})  "
              f"iou={vl['mean_iou']:.3f}  "
              f"sev_exact={vl['sev_exact']:.3f}  sev_adj={vl['sev_adj']:.3f}")

        history.append({"epoch":epoch+1, **{f"tr_{k}":v for k,v in tr.items()},
                        **{k:v for k,v in vl.items() if k not in ('sev_true','sev_pred')}})

        if vl["mean_dice"] > best_dice:
            best_dice = vl["mean_dice"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch+1,
                "metrics": {k:v for k,v in vl.items() if k not in ('sev_true','sev_pred')},
            }, cfg.CKPT_DIR / "fracatlas_seg_sev.pt")

            with open(cfg.CKPT_DIR / "fracatlas_eval.json", "w") as f:
                json.dump({
                    "metrics": {k:v for k,v in vl.items() if k not in ('sev_true','sev_pred')},
                    "severity_confusion": confusion_matrix(
                        vl["sev_true"], vl["sev_pred"], labels=[0,1,2,3]).tolist(),
                }, f, indent=2)
            print(f"     → fracatlas_seg_sev.pt (Dice={best_dice:.3f})")

        torch.cuda.empty_cache(); gc.collect()

    pd.DataFrame(history).to_csv(cfg.CKPT_DIR / "fracatlas_history.csv", index=False)
    print(f"\n[DONE] Best Dice: {best_dice:.3f}")


if __name__ == "__main__":
    main()

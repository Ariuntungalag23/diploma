"""
=============================================================================
TRAIN 3/3: GRAZPEDWRI-DX HEALING REGRESSION
=============================================================================
Зорилго:
  Рентген зургаас + клиник метаөгөгдөл (нас, хүйс, байршил)-аас
  эдгэрэх хугацаа (долоо хоног) таамаглана.

Архитектур:
  EfficientNet-B3 encoder (FracAtlas-аас pretrained жинтэйгээр эхлүүлж
                            болно, эсвэл ImageNet-ээс)
    └→ HealingHead (encoder feat + meta → weeks regression)

Шошго:
  GRAZPEDWRI-DX-д бодит healing хугацаа байхгүй учир Molitoris (2024),
  Fini (2023), StatPearls NBK551678-д суурилсан синтетик шошго.

Нөөц:
  ~30-45 мин сургалт, ~3 GB VRAM
=============================================================================
"""

import os, sys, time, json, gc, re, random
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent / "fracture_thesis"
    FILTERED_DIR = PROJECT_ROOT / "filtered_lists"
    CKPT_DIR     = PROJECT_ROOT / "checkpoints"

    # Encoder warm-start: FracAtlas-аас сургагдсан encoder ашиглах
    PRETRAINED_ENCODER = CKPT_DIR / "fracatlas_seg_sev.pt"  # optional

    IMG_SIZE     = 224
    BATCH_SIZE   = 32
    EPOCHS       = 15
    LR_ENCODER   = 1e-5    # encoder бараг хөлдөөнө
    LR_HEAD      = 3e-4
    WEIGHT_DECAY = 1e-4

    USE_AMP      = True
    NUM_WORKERS  = 8
    SEED         = 42

    HEAL_META_DIM = 4
    LOCATIONS = ['wrist','forearm','humerus','tibia','ankle','clavicle','femur']

cfg = Config()
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# СИНТЕТИК HEALING ШОШГО (Molitoris 2024, Fini 2023, StatPearls)
# =============================================================================
def synthesize_healing_time(age, sex, severity, location='wrist',
                             fracture_type='simple'):
    if   age < 6:  base = 2.5
    elif age < 12: base = 3.5
    elif age < 18: base = 5.0
    elif age < 40: base = 7.0
    elif age < 60: base = 9.0
    else:          base = 11.0

    severity_factor = 1.0 + 0.35 * severity
    type_factor = {
        'simple':1.0,'transverse':1.0,'oblique':1.1,'spiral':1.15,
        'comminuted':1.4,'compression':1.2,'greenstick':0.85,'open':1.5,
    }.get(fracture_type, 1.0)
    location_factor = {
        'wrist':1.0,'forearm':1.05,'humerus':1.2,'femur':1.5,
        'tibia':1.4,'ankle':1.1,'clavicle':0.9,
    }.get(location, 1.0)
    sex_factor = 1.0 if sex == 'M' else 1.03
    noise = np.clip(np.random.normal(1.0, 0.12), 0.7, 1.4)

    weeks = base * severity_factor * type_factor * location_factor * sex_factor * noise
    return max(1.0, weeks)


# =============================================================================
# HEALING HEAD
# =============================================================================
class HealingHead(nn.Module):
    """Encoder feat + meta → weeks regression (Gaussian NLL)."""
    def __init__(self, in_dim=384, meta_dim=3, drop=0.2):
        super().__init__()
        self.meta_emb = nn.Sequential(
            nn.Linear(meta_dim, 16), nn.ReLU(True),
        )
        cdim = in_dim + 16  # encoder + meta

        self.reg = nn.Sequential(
            nn.Linear(cdim, 256),
            nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, 1),
        )
        self.log_var = nn.Sequential(
            nn.Linear(cdim, 64), nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, feat, meta):
        g = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        c = torch.cat([g, self.meta_emb(meta)], dim=1)
        return self.reg(c), self.log_var(c)


# =============================================================================
# MODEL (GRAZPED only)
# =============================================================================
class HealingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3", pretrained=True,
            features_only=True, out_indices=[4])
        self.heal_head = HealingHead(in_dim=384, meta_dim=cfg.HEAL_META_DIM)

    def forward(self, x, meta):
        feat = self.encoder(x)[0]
        return self.heal_head(feat, meta)


def load_pretrained_encoder(model, ckpt_path):
    """FracAtlas сургагдсан encoder-аас warm-start хийх."""
    if not ckpt_path.exists():
        print(f"  [INFO] {ckpt_path} байхгүй — ImageNet жингээр л эхэлнэ")
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items()
                 if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=False)
    print(f"  [OK] FracAtlas encoder-ыг warm-start хийсэн ({len(enc_state)} key)")
    return True


# =============================================================================
# LOSS
# =============================================================================
def gaussian_nll_loss(pred, target, log_var):
    precision = torch.exp(-log_var)
    return 0.5 * (precision * (pred - target).pow(2) + log_var).mean()


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


class HealingDataset(Dataset):
    """GRAZPEDWRI-DX + синтетик healing шошго."""

    def __init__(self, csv_path, split="train"):
        df = pd.read_csv(csv_path)
        self.df = df.reset_index(drop=True)
        self.tf = TRAIN_TF if split == "train" else VAL_TF
        self._prepare_synthetic_labels()
        print(f"  [GRAZPED {split}] {len(self.df)} зураг")

    def _parse_filename(self, fname):
        m = re.search(r'([MF])(\d{3})', str(fname))
        if m:
            return m.group(1), int(m.group(2))
        return random.choice(['M','F']), random.randint(5, 17)

    def _prepare_synthetic_labels(self):
        rng = np.random.RandomState(cfg.SEED)
        records = []
        for _, row in self.df.iterrows():
            fname = row.get('filename', os.path.basename(str(row['path'])))
            sex, age = self._parse_filename(fname)
            ftype = rng.choice(['transverse','oblique','spiral','comminuted','greenstick'])
            loc   = rng.choice(cfg.LOCATIONS)
            sev   = int(rng.randint(0, 4))
            weeks = synthesize_healing_time(
                age=age, sex=sex, severity=sev,
                location=loc, fracture_type=ftype
            )
            records.append({
                'sex':sex, 'age':age, 'fracture_type':ftype,
                'location':loc, 'severity':sev, 'healing_weeks':weeks,
            })
        for col in records[0].keys():
            self.df[col] = [r[col] for r in records]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        aug = self.tf(image=img)

        age_norm = row['age'] / 100.0
        sex_bin  = 1.0 if row['sex'] == 'M' else 0.0
        loc_id   = float(cfg.LOCATIONS.index(row['location'])) / len(cfg.LOCATIONS)
        sev_norm = float(row['severity']) / 3.0
        meta = torch.tensor([age_norm, sex_bin, loc_id, sev_norm], dtype=torch.float32)

        return {
            "image": aug["image"],
            "meta":  meta,
            "heal":  torch.tensor(float(row['healing_weeks']), dtype=torch.float32),
        }


# =============================================================================
# TRAIN / VAL
# =============================================================================
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0; n = 0
    pbar = tqdm(loader, desc=f"Ep{epoch+1}/{cfg.EPOCHS} train", ascii=True)
    for batch in pbar:
        x      = batch["image"].to(DEVICE, non_blocking=True)
        meta   = batch["meta"].to(DEVICE)
        y      = batch["heal"].to(DEVICE).unsqueeze(1)

        with autocast(enabled=cfg.USE_AMP):
            pred, log_var = model(x, meta)
            loss = gaussian_nll_loss(pred, y, log_var)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.3f}")
    return total_loss / n


@torch.no_grad()
def validate(model, loader, epoch):
    model.eval()
    preds, targets = [], []
    for batch in tqdm(loader, desc=f"Ep{epoch+1} val", ascii=True):
        x    = batch["image"].to(DEVICE)
        meta = batch["meta"].to(DEVICE)
        y    = batch["heal"].to(DEVICE).unsqueeze(1)

        with autocast(enabled=cfg.USE_AMP):
            pred, _ = model(x, meta)
        preds.append(pred.cpu().float().numpy().flatten())
        targets.append(y.cpu().float().numpy().flatten())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "MAE":  float(mean_absolute_error(targets, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
        "R2":   float(r2_score(targets, preds)),
        "preds":   preds.tolist(),
        "targets": targets.tolist(),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("TRAIN 3/3: GRAZPEDWRI-DX HEALING REGRESSION")
    print("="*70)
    print(f"  Device: {DEVICE}")
    print(f"  GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    csv = cfg.FILTERED_DIR / "grazpedwri_final.csv"
    if not csv.exists():
        print(f"\n[ERR] {csv} байхгүй"); sys.exit(1)

    df = pd.read_csv(csv).sample(frac=1, random_state=cfg.SEED).reset_index(drop=True)
    n_tr = int(0.8 * len(df))
    tr_csv = cfg.PROJECT_ROOT / "_grazped_train.csv"
    va_csv = cfg.PROJECT_ROOT / "_grazped_val.csv"
    df.iloc[:n_tr].to_csv(tr_csv, index=False)
    df.iloc[n_tr:].to_csv(va_csv, index=False)

    train_ds = HealingDataset(tr_csv, "train")
    val_ds   = HealingDataset(va_csv, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    model = HealingModel().to(DEVICE)

    # Encoder warm-start (хэрэв FracAtlas сургалт дууссан бол)
    load_pretrained_encoder(model, cfg.PRETRAINED_ENCODER)

    if hasattr(model.encoder, "set_grad_checkpointing"):
        model.encoder.set_grad_checkpointing(enable=True)

    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total : {total/1e6:.1f}M  |  Trainable: {train/1e6:.1f}M")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg.LR_ENCODER},
        {"params": model.heal_head.parameters(),  "lr": cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    best_mae = float('inf'); history = []
    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        vl = validate(model, val_loader, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"\n  Ep{epoch+1:02d} ({elapsed:.0f}s) | tr_loss={tr_loss:.3f} | "
              f"MAE={vl['MAE']:.3f}w  RMSE={vl['RMSE']:.3f}w  R²={vl['R2']:.3f}")

        history.append({"epoch":epoch+1, "tr_loss":tr_loss,
                        **{k:v for k,v in vl.items() if k not in ('preds','targets')}})

        if vl["MAE"] < best_mae:
            best_mae = vl["MAE"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch+1,
                "metrics": {k:v for k,v in vl.items() if k not in ('preds','targets')},
            }, cfg.CKPT_DIR / "grazped_healing.pt")

            with open(cfg.CKPT_DIR / "grazped_eval.json", "w") as f:
                json.dump({k:v for k,v in vl.items() if k != 'preds' and k != 'targets'}, f, indent=2)
            print(f"     → grazped_healing.pt (MAE={best_mae:.3f}w)")

        torch.cuda.empty_cache(); gc.collect()

    pd.DataFrame(history).to_csv(cfg.CKPT_DIR / "grazped_history.csv", index=False)
    print(f"\n[DONE] Best MAE: {best_mae:.3f} долоо хоног")


if __name__ == "__main__":
    main()

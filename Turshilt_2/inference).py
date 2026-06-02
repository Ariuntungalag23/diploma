"""
=============================================================================
END-TO-END INFERENCE — SINGLE IMAGE → ALL 3 MODELS
=============================================================================
Flow (1 зураг):
  shared EfficientNet-B3 encoder (1 forward pass)
    ├─ [1] MURA head        → fracture / normal  (binary)
    ├─ [2] FracAtlas heads  → segmentation mask + severity grade (0-3)
    └─ [3] GRAZPED head     → healing weeks (regression + uncertainty)

Хэрэглэх:
  python inference_e2e.py --image path/to/xray.jpg
  python inference_e2e.py --image xray.jpg --age 12 --sex F --location wrist
  python inference_e2e.py --image xray.jpg --no_seg   # SAM-гүй хурдан горим
=============================================================================
"""

import os, sys, json, math, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT  = Path(__file__).resolve().parent / "fracture_thesis"
CKPT_DIR      = PROJECT_ROOT / "checkpoints"
SAM_CKPT      = PROJECT_ROOT / "sam_vit_b.pth"

MURA_CKPT  = CKPT_DIR / "mura_classifier.pt"
FRAC_CKPT  = CKPT_DIR / "fracatlas_seg_sev.pt"
HEAL_CKPT  = CKPT_DIR / "grazped_healing.pt"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
SAM_SIZE  = 1024
LOCATIONS = ['wrist', 'forearm', 'humerus', 'tibia', 'ankle', 'clavicle', 'femur']
SEVERITY_LABELS = {
    0: "Grade 0 — Minimal",
    1: "Grade 1 — Mild",
    2: "Grade 2 — Moderate",
    3: "Grade 3 — Severe",
}


# =============================================================================
# TRANSFORM
# =============================================================================
INFER_TF = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def load_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    return INFER_TF(image=img)["image"].unsqueeze(0).to(DEVICE)


# =============================================================================
# SHARED ENCODER  (1 instance, shared across all heads)
# =============================================================================
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False,
            features_only=True, out_indices=[4])

    def forward(self, x):
        return self.backbone(x)[0]   # (B, 384, 7, 7)


# =============================================================================
# HEAD 1 — MURA binary classifier
# =============================================================================
class MURAHead(nn.Module):
    def __init__(self, in_dim=384, drop=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 2),
        )

    def forward(self, feat):
        return self.head(feat)


# =============================================================================
# HEAD 2a — LoRA helpers
# =============================================================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4, alpha=4):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.lora_A.T) @ self.lora_B.T)


def inject_lora(module, rank=4, alpha=4, target_names=("q_proj", "v_proj")):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += inject_lora(child, rank, alpha, target_names)
    return n


# =============================================================================
# HEAD 2b — SAM segmentation (optional, needs sam_vit_b.pth)
# =============================================================================
class SAMSegHead(nn.Module):
    def __init__(self, rank=4, alpha=4):
        super().__init__()
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
        self.image_encoder  = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder   = sam.mask_decoder
        for p in self.parameters():
            p.requires_grad = False
        inject_lora(self.mask_decoder, rank=rank, alpha=alpha)
        self.register_buffer("pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, x_norm):
        mean = torch.tensor([0.485,0.456,0.406], device=x_norm.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x_norm.device).view(1,3,1,1)
        raw  = F.interpolate((x_norm * std + mean) * 255.,
                             size=(SAM_SIZE, SAM_SIZE),
                             mode="bilinear", align_corners=False)
        with torch.no_grad():
            emb = self.image_encoder((raw - self.pixel_mean) / self.pixel_std)

        cx = SAM_SIZE // 2; half = SAM_SIZE // 4
        bbox = torch.tensor([[cx-half, cx-half, cx+half, cx+half]],
                             device=x_norm.device, dtype=torch.float32)
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(
                points=None, boxes=bbox, masks=None)
        low_res, _ = self.mask_decoder(
            image_embeddings=emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return F.interpolate(low_res, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)


# =============================================================================
# HEAD 2c — Severity ordinal (uses encoder feat + mask)
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
        m = torch.sigmoid(mask_logits).mean(dim=(1, 2, 3)).unsqueeze(1)
        return self.fc(torch.cat([f, m], dim=1))


# =============================================================================
# HEAD 3 — Healing regression (uses encoder feat + meta)
# =============================================================================
class HealingHead(nn.Module):
    def __init__(self, in_dim=384, meta_dim=4, drop=0.2):
        super().__init__()
        self.meta_emb = nn.Sequential(nn.Linear(meta_dim, 16), nn.ReLU(True))
        cdim = in_dim + 16
        self.reg = nn.Sequential(
            nn.Linear(cdim, 256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 1),
        )
        self.log_var = nn.Sequential(
            nn.Linear(cdim, 64), nn.ReLU(True), nn.Linear(64, 1))

    def forward(self, feat, meta):
        g = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        c = torch.cat([g, self.meta_emb(meta)], dim=1)
        return self.reg(c), self.log_var(c)


# =============================================================================
# UNIFIED E2E MODEL
# =============================================================================
class FractureE2E(nn.Module):
    """
    Single forward pass → all predictions.
    Encoder runs ONCE; feat tensor is reused by every head.
    """
    def __init__(self, use_seg=True):
        super().__init__()
        self.encoder   = SharedEncoder()
        self.mura_head = MURAHead(in_dim=384)
        self.sev_head  = SeverityHead(in_dim=384, n_grades=4)
        self.heal_head = HealingHead(in_dim=384, meta_dim=4)

        self.use_seg = use_seg
        self.seg_head = SAMSegHead(rank=4, alpha=4) if use_seg else None

    def forward(self, x, meta=None):
        # ── Shared encoder (single pass) ──
        feat = self.encoder(x)                          # (B, 384, 7, 7)

        # ── Head 1: MURA ──
        cls_logits = self.mura_head(feat)               # (B, 2)

        # ── Head 2: Seg + Severity ──
        if self.use_seg and self.seg_head is not None:
            mask_logits = self.seg_head(x)              # (B, 1, H, W)
        else:
            mask_logits = torch.zeros(
                x.shape[0], 1, IMG_SIZE, IMG_SIZE, device=x.device)

        sev_logits = self.sev_head(feat, mask_logits)   # (B, 3)

        # ── Head 3: Healing ──
        if meta is None:
            meta = torch.zeros(x.shape[0], 4, device=x.device)
        heal_pred, heal_log_var = self.heal_head(feat, meta)  # (B,1), (B,1)

        return {
            "cls":          cls_logits,
            "mask":         mask_logits,
            "sev":          sev_logits,
            "heal_pred":    heal_pred,
            "heal_log_var": heal_log_var,
        }


# =============================================================================
# CHECKPOINT LOADER
# =============================================================================
def _filter_keys(state, prefix):
    """Extract sub-model weights from a full model checkpoint."""
    return {k[len(prefix):]: v
            for k, v in state.items() if k.startswith(prefix)}


def load_e2e_model(use_seg=True):
    """
    Build FractureE2E and load each head's weights from its own checkpoint.
    The encoder weights come from MURA (trained first); FracAtlas / GRAZPED
    encoder weights are also loaded and averaged to give the best shared init.
    """
    model = FractureE2E(use_seg=use_seg).to(DEVICE)

    missing = []
    for p, name in [(MURA_CKPT, "MURA"), (FRAC_CKPT, "FracAtlas"), (HEAL_CKPT, "GRAZPED")]:
        if not p.exists():
            missing.append(f"  [WARN] {name} checkpoint not found: {p}")
    if missing:
        print("\n".join(missing))

    # ── MURA: encoder + mura_head ──
    if MURA_CKPT.exists():
        mura_state = torch.load(MURA_CKPT, map_location=DEVICE)["model"]
        # encoder keys in MURA checkpoint are under "encoder."
        enc_w = _filter_keys(mura_state, "encoder.")
        model.encoder.backbone.load_state_dict(enc_w, strict=False)
        # head keys are under "head."
        head_w = _filter_keys(mura_state, "head.")
        model.mura_head.head.load_state_dict(head_w, strict=False)
        print("  [OK] MURA  encoder + classifier head loaded")

    # ── FracAtlas: encoder (averaged), seg_head, sev_head ──
    if FRAC_CKPT.exists():
        frac_state = torch.load(FRAC_CKPT, map_location=DEVICE)["model"]

        # Average encoder weights with MURA encoder for better shared init
        frac_enc_w = _filter_keys(frac_state, "encoder.")
        if frac_enc_w:
            cur = model.encoder.backbone.state_dict()
            for k in cur:
                if k in frac_enc_w and cur[k].shape == frac_enc_w[k].shape:
                    cur[k] = (cur[k] + frac_enc_w[k]) / 2.0
            model.encoder.backbone.load_state_dict(cur, strict=False)
            print("  [OK] FracAtlas encoder averaged into shared encoder")

        # seg_head  (stored as "seg_head." in FracAtlas checkpoint)
        if use_seg and model.seg_head is not None:
            seg_w = _filter_keys(frac_state, "seg_head.")
            model.seg_head.load_state_dict(seg_w, strict=False)
            print("  [OK] FracAtlas SAM seg head loaded")

        # sev_head
        sev_w = _filter_keys(frac_state, "sev_head.")
        model.sev_head.load_state_dict(sev_w, strict=False)
        print("  [OK] FracAtlas severity head loaded")

    # ── GRAZPED: encoder (averaged), heal_head ──
    if HEAL_CKPT.exists():
        heal_state = torch.load(HEAL_CKPT, map_location=DEVICE)["model"]

        graz_enc_w = _filter_keys(heal_state, "encoder.")
        if graz_enc_w:
            cur = model.encoder.backbone.state_dict()
            for k in cur:
                if k in graz_enc_w and cur[k].shape == graz_enc_w[k].shape:
                    cur[k] = (cur[k] * 2 + graz_enc_w[k]) / 3.0
            model.encoder.backbone.load_state_dict(cur, strict=False)
            print("  [OK] GRAZPED encoder averaged into shared encoder")

        heal_w = _filter_keys(heal_state, "heal_head.")
        model.heal_head.load_state_dict(heal_w, strict=False)
        print("  [OK] GRAZPED healing head loaded")

    model.eval()
    return model


# =============================================================================
# INFERENCE
# =============================================================================
def build_meta(age, sex, location, severity):
    age_norm = age / 100.0
    sex_bin  = 1.0 if str(sex).upper() == "M" else 0.0
    loc_id   = float(LOCATIONS.index(location)) / len(LOCATIONS) \
               if location in LOCATIONS else 0.0
    sev_norm = float(severity) / 3.0
    return torch.tensor([[age_norm, sex_bin, loc_id, sev_norm]],
                        dtype=torch.float32, device=DEVICE)


@torch.inference_mode()
def run(model, image_path, age=25, sex="M", location="wrist", save_mask=False):
    x    = load_image(image_path)

    # First pass without meta to get severity for meta vector
    out  = model(x, meta=None)

    # ── Classification ──
    cls_probs  = F.softmax(out["cls"], dim=1)[0].cpu().tolist()
    is_fracture = int(out["cls"].argmax(dim=1).item()) == 1

    # ── Severity ──
    severity = int((torch.sigmoid(out["sev"]) > 0.5).sum(dim=1).item())

    # ── Second pass with proper meta (severity now known) ──
    meta = build_meta(age, sex, location, severity)
    out  = model(x, meta=meta)

    heal_weeks = float(out["heal_pred"].item())
    heal_std   = float(torch.exp(0.5 * out["heal_log_var"]).item())

    # ── Mask ──
    mask_prob   = torch.sigmoid(out["mask"])[0, 0].cpu().numpy()
    mask_binary = (mask_prob > 0.5).astype(np.uint8)
    coverage    = float(mask_binary.mean())

    if save_mask and is_fracture:
        mp = Path(image_path).with_name(Path(image_path).stem + "_mask.png")
        Image.fromarray(mask_binary * 255).save(mp)

    return {
        "image":   str(image_path),
        "patient": {"age": age, "sex": sex, "location": location},
        "classification": {
            "fracture":   is_fracture,
            "confidence": round(max(cls_probs), 4),
            "prob_normal":   round(cls_probs[0], 4),
            "prob_fracture": round(cls_probs[1], 4),
        },
        "segmentation": {
            "mask_coverage":  round(coverage, 4),
            "severity_grade": severity,
            "severity_label": SEVERITY_LABELS.get(severity, f"Grade {severity}"),
        } if is_fracture else None,
        "healing": {
            "healing_weeks":  round(heal_weeks, 1),
            "healing_range":  [round(max(0, heal_weeks - 1.96*heal_std), 1),
                               round(heal_weeks + 1.96*heal_std, 1)],
            "uncertainty_std": round(heal_std, 2),
        } if is_fracture else None,
    }


# =============================================================================
# PRETTY PRINT
# =============================================================================
def print_result(r):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  IMAGE   : {Path(r['image']).name}")
    pt = r["patient"]
    print(f"  PATIENT : age={pt['age']}  sex={pt['sex']}  location={pt['location']}")
    print(sep)

    cls = r["classification"]
    tag = "✔ FRACTURE DETECTED" if cls["fracture"] else "✘ No fracture"
    print(f"\n  [1] DETECTION    {tag}  (conf={cls['confidence']:.1%})")
    print(f"      normal={cls['prob_normal']:.3f}   fracture={cls['prob_fracture']:.3f}")

    if not cls["fracture"]:
        print(f"\n  [2] SEGMENTATION —  (skipped — no fracture)")
        print(f"  [3] HEALING      —  (skipped — no fracture)")
        print(sep); return

    seg = r["segmentation"]
    print(f"\n  [2] SEGMENTATION  coverage={seg['mask_coverage']:.1%}")
    print(f"      SEVERITY      {seg['severity_label']}")

    h = r["healing"]
    lo, hi = h["healing_range"]
    print(f"\n  [3] HEALING EST.  {h['healing_weeks']} weeks")
    print(f"      95% CI        {lo} – {hi} weeks   (σ={h['uncertainty_std']})")
    print(sep)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fracture X-ray — single image → all 3 models")
    p.add_argument("--image",    required=True, help="X-ray image path")
    p.add_argument("--age",      type=int,   default=25)
    p.add_argument("--sex",      type=str,   default="M", choices=["M","F"])
    p.add_argument("--location", type=str,   default="wrist", choices=LOCATIONS)
    p.add_argument("--save_mask",action="store_true")
    p.add_argument("--no_seg",   action="store_true",
                   help="Skip SAM seg head (no sam_vit_b.pth needed)")
    p.add_argument("--output",   type=str,   default="",
                   help="Save JSON result to this path")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("FRACTURE E2E PIPELINE  —  loading models...")
    print("="*60)
    model = load_e2e_model(use_seg=not args.no_seg)

    result = run(
        model, args.image,
        age=args.age, sex=args.sex,
        location=args.location, save_mask=args.save_mask,
    )
    print_result(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()
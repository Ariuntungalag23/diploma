"""
=============================================================================
END-TO-END INFERENCE PIPELINE
=============================================================================
3 загварын дараалсан таамаглал:
  1. MURA Classifier     → хугарал байгаа эсэх (binary)
  2. FracAtlas Seg+Sev   → хугарлын маск + хүндрэлийн зэрэг (Grade 0-3)
  3. GRAZPED Healing     → эдгэрэх хугацаа (долоо хоног)

Хэрэглэх:
  python inference_pipeline.py --image path/to/xray.jpg
  python inference_pipeline.py --image xray.jpg --age 12 --sex M --location wrist
  python inference_pipeline.py --folder path/to/images/ --output results.json
=============================================================================
"""

import os, sys, json, math, re, argparse, hashlib, warnings
import pathlib
from pathlib import Path

# ── Windows fix: checkpoints saved on Linux contain PosixPath ──
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

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
PROJECT_ROOT  = Path(__file__).resolve().parent
CKPT_DIR      = PROJECT_ROOT
SAM_CKPT      = PROJECT_ROOT / "sam_vit_b.pth"
SAM_EMB_CACHE = PROJECT_ROOT / "sam_emb_cache"

MURA_CKPT    = CKPT_DIR / "mura_classifier.pt"
FRAC_CKPT    = CKPT_DIR / "fracatlas_seg_sev.pt"
HEAL_CKPT    = CKPT_DIR / "grazped_healing.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
SAM_SIZE  = 1024

LOCATIONS = ['wrist', 'forearm', 'humerus', 'tibia', 'ankle', 'clavicle', 'femur']
SEVERITY_LABELS = {0: "Grade 0 (Minimal)", 1: "Grade 1 (Mild)",
                   2: "Grade 2 (Moderate)", 3: "Grade 3 (Severe)"}


# =============================================================================
# SHARED TRANSFORMS
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
# MODEL DEFINITIONS (mirrors train scripts exactly)
# =============================================================================

# ── 1. MURA Classifier ───────────────────────────────────────────────────────
class MURAClassifier(nn.Module):
    def __init__(self, n_classes=2, drop=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3", pretrained=False,
            features_only=True, out_indices=[4])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(384, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x)[0])


# ── 2a. LoRA helpers ──────────────────────────────────────────────────────────
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


# ── 2b. SAM Seg Head ──────────────────────────────────────────────────────────
class SAMSegHead(nn.Module):
    def __init__(self, sam_ckpt, rank=4, alpha=4):
        super().__init__()
        from segment_anything import sam_model_registry
        sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
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

    def _encode(self, x_norm):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x_norm.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x_norm.device).view(1,3,1,1)
        raw  = (x_norm * std + mean) * 255.0
        raw  = F.interpolate(raw, size=(SAM_SIZE, SAM_SIZE),
                             mode="bilinear", align_corners=False)
        with torch.no_grad():
            return self.image_encoder((raw - self.pixel_mean) / self.pixel_std)

    def forward(self, x_norm, bboxes=None):
        B = x_norm.shape[0]
        image_emb = self._encode(x_norm)

        if bboxes is None:
            cx = SAM_SIZE // 2; half = SAM_SIZE // 4
            bboxes_sam = torch.tensor(
                [[cx - half, cx - half, cx + half, cx + half]],
                device=x_norm.device).float().repeat(B, 1)
        else:
            bboxes_sam = bboxes.float() * (SAM_SIZE / IMG_SIZE)

        low_res_masks = []
        image_pe = self.prompt_encoder.get_dense_pe()
        for i in range(B):
            with torch.no_grad():
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=None, boxes=bboxes_sam[i:i+1], masks=None)
            low_res, _ = self.mask_decoder(
                image_embeddings=image_emb[i:i+1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            low_res_masks.append(low_res)
        low_res = torch.cat(low_res_masks, dim=0)
        return F.interpolate(low_res, size=(IMG_SIZE, IMG_SIZE),
                             mode="bilinear", align_corners=False)


# ── 2c. Severity Head ─────────────────────────────────────────────────────────
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


# ── 2d. FracAtlas Model ───────────────────────────────────────────────────────
class FracAtlasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b3", pretrained=False,
            features_only=True, out_indices=[4])
        self.seg_head = SAMSegHead(SAM_CKPT, rank=4, alpha=4)
        self.sev_head = SeverityHead(in_dim=384, n_grades=4)

    def forward(self, x):
        feat = self.encoder(x)[0]
        mask = self.seg_head(x)
        sev  = self.sev_head(feat, mask)
        return feat, mask, sev


# ── 3. Healing Model ──────────────────────────────────────────────────────────
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


class HealingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = timm.create_model(
            "efficientnet_b3", pretrained=False,
            features_only=True, out_indices=[4])
        self.heal_head = HealingHead(in_dim=384, meta_dim=4)

    def forward(self, x, meta):
        feat = self.encoder(x)[0]
        return self.heal_head(feat, meta)


# =============================================================================
# MODEL LOADER
# =============================================================================
def load_mura(ckpt_path):
    model = MURAClassifier().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [OK] MURA classifier  — {ckpt_path.name}")
    return model


def load_fracatlas(ckpt_path):
    if not SAM_CKPT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CKPT}\n"
            "Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    model = FracAtlasModel().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [OK] FracAtlas seg+sev — {ckpt_path.name}")
    return model


def load_healing(ckpt_path):
    model = HealingModel().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [OK] Healing regressor — {ckpt_path.name}")
    return model


# =============================================================================
# INFERENCE STEPS
# =============================================================================
@torch.inference_mode()
def step1_classify(model, x):
    """Returns: {'fracture': bool, 'confidence': float, 'probs': [p_normal, p_fracture]}"""
    logits = model(x)
    probs  = F.softmax(logits, dim=1)[0].cpu().tolist()
    label  = int(logits.argmax(dim=1).item())
    return {
        "fracture":   bool(label == 1),
        "confidence": float(max(probs)),
        "probs":      {"normal": round(probs[0], 4), "fracture": round(probs[1], 4)},
    }


@torch.inference_mode()
def step2_segment_severity(model, x):
    """Returns: {'severity_grade': int, 'severity_label': str,
                  'mask_coverage': float, 'mask': np.ndarray (H,W)}"""
    feat, mask_logits, sev_logits = model(x)
    grade = int((torch.sigmoid(sev_logits) > 0.5).sum(dim=1).item())
    mask_prob    = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
    mask_binary  = (mask_prob > 0.5).astype(np.uint8)
    coverage     = float(mask_binary.mean())
    return {
        "severity_grade": grade,
        "severity_label": SEVERITY_LABELS.get(grade, f"Grade {grade}"),
        "mask_coverage":  round(coverage, 4),
        "mask":           mask_binary,
    }


@torch.inference_mode()
def step3_healing(model, x, age, sex, location, severity):
    """Returns: {'healing_weeks': float, 'healing_range': [lo, hi], 'uncertainty': float}"""
    age_norm = age / 100.0
    sex_bin  = 1.0 if str(sex).upper() == "M" else 0.0
    loc_id   = float(LOCATIONS.index(location)) / len(LOCATIONS) \
               if location in LOCATIONS else 0.0
    sev_norm = float(severity) / 3.0
    meta = torch.tensor([[age_norm, sex_bin, loc_id, sev_norm]],
                        dtype=torch.float32, device=DEVICE)
    pred, log_var = model(x, meta)
    weeks = float(pred.item())
    std   = float(torch.exp(0.5 * log_var).item())
    return {
        "healing_weeks": round(weeks, 1),
        "healing_range": [round(max(0, weeks - 1.96*std), 1),
                          round(weeks + 1.96*std, 1)],
        "uncertainty_std": round(std, 2),
    }


# =============================================================================
# FULL PIPELINE
# =============================================================================
class FracturePipeline:
    def __init__(self, load_seg=True):
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)

        for ckpt, name in [(MURA_CKPT, "MURA"), (FRAC_CKPT, "FracAtlas"), (HEAL_CKPT, "GRAZPED")]:
            if not ckpt.exists():
                print(f"  [WARN] {name} checkpoint not found: {ckpt}")

        self.mura    = load_mura(MURA_CKPT)
        self.frac    = load_fracatlas(FRAC_CKPT) if load_seg else None
        self.healing = load_healing(HEAL_CKPT)
        print("="*60 + "\n")

    def predict(self, image_path, age=25, sex="M", location="wrist",
                save_mask=False):
        """
        Run full pipeline on a single image.

        Args:
            image_path : str | Path — input X-ray
            age        : int  — patient age (years)
            sex        : str  — 'M' or 'F'
            location   : str  — anatomical site (see LOCATIONS)
            save_mask  : bool — save segmentation mask PNG next to image

        Returns:
            dict with all predictions + metadata
        """
        path = Path(image_path)
        x    = load_image(path)

        result = {
            "image": str(path),
            "patient": {"age": age, "sex": sex, "location": location},
        }

        # ── Step 1: Fracture detection ──
        cls = step1_classify(self.mura, x)
        result["classification"] = cls

        if not cls["fracture"]:
            result["classification"]["note"] = "No fracture detected — pipeline stops here."
            result["segmentation"] = None
            result["healing"]      = None
            return result

        # ── Step 2: Segmentation + severity ──
        if self.frac is not None:
            seg = step2_segment_severity(self.frac, x)
            severity = seg["severity_grade"]
            mask_arr = seg.pop("mask")
            result["segmentation"] = seg

            if save_mask:
                mask_path = path.with_name(path.stem + "_mask.png")
                Image.fromarray(mask_arr * 255).save(mask_path)
                result["segmentation"]["mask_saved"] = str(mask_path)
        else:
            severity = 1  # default if seg model not loaded
            result["segmentation"] = {"note": "Seg model not loaded"}

        # ── Step 3: Healing regression ──
        heal = step3_healing(self.healing, x,
                             age=age, sex=sex,
                             location=location, severity=severity)
        result["healing"] = heal

        return result

    def predict_batch(self, image_paths, age=25, sex="M", location="wrist"):
        """Run pipeline on a list of images."""
        results = []
        for i, p in enumerate(image_paths, 1):
            print(f"  [{i}/{len(image_paths)}] {Path(p).name}")
            try:
                r = self.predict(p, age=age, sex=sex, location=location)
            except Exception as e:
                r = {"image": str(p), "error": str(e)}
            results.append(r)
        return results


# =============================================================================
# PRETTY PRINT
# =============================================================================
def print_result(r):
    print("\n" + "─"*60)
    print(f"  Image : {Path(r['image']).name}")
    if "error" in r:
        print(f"  ERROR : {r['error']}")
        return

    meta = r.get("patient", {})
    print(f"  Patient: age={meta.get('age')}  sex={meta.get('sex')}  "
          f"location={meta.get('location')}")

    cls = r.get("classification", {})
    status = "FRACTURE DETECTED" if cls.get("fracture") else "No fracture"
    conf   = cls.get("confidence", 0)
    print(f"\n  [1] Classification : {status}  (conf={conf:.1%})")
    print(f"      Probs → normal={cls['probs']['normal']:.3f}  "
          f"fracture={cls['probs']['fracture']:.3f}")

    if not cls.get("fracture"):
        print("  → Pipeline stopped (no fracture).")
        return

    seg = r.get("segmentation")
    if seg and "severity_grade" in seg:
        print(f"\n  [2] Segmentation   : coverage={seg['mask_coverage']:.1%}")
        print(f"      Severity       : {seg['severity_label']}")

    heal = r.get("healing")
    if heal:
        lo, hi = heal["healing_range"]
        print(f"\n  [3] Healing est.   : {heal['healing_weeks']} weeks "
              f"(95% CI: {lo}–{hi} wk,  σ={heal['uncertainty_std']})")
    print("─"*60)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fracture X-ray — end-to-end inference pipeline")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",  type=str, help="Single X-ray image path")
    grp.add_argument("--folder", type=str, help="Folder of images")
    p.add_argument("--age",      type=int,   default=25,
                   help="Patient age (default: 25)")
    p.add_argument("--sex",      type=str,   default="M",
                   choices=["M","F"], help="Sex: M or F (default: M)")
    p.add_argument("--location", type=str,   default="wrist",
                   choices=LOCATIONS, help="Fracture site (default: wrist)")
    p.add_argument("--output",   type=str,   default="",
                   help="Save JSON results to this path")
    p.add_argument("--save_mask", action="store_true",
                   help="Save segmentation mask PNG alongside each image")
    p.add_argument("--no_seg",   action="store_true",
                   help="Skip FracAtlas model (faster, no SAM dependency)")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = FracturePipeline(load_seg=not args.no_seg)

    if args.image:
        result = pipeline.predict(
            args.image, age=args.age, sex=args.sex,
            location=args.location, save_mask=args.save_mask)
        print_result(result)
        results = [result]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        paths = [p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts]
        print(f"  Found {len(paths)} images in {args.folder}")
        results = pipeline.predict_batch(
            paths, age=args.age, sex=args.sex, location=args.location)
        for r in results:
            print_result(r)

    if args.output:
        # Remove numpy arrays before serialising
        clean = []
        for r in results:
            cr = {k: v for k, v in r.items() if k != "mask"}
            clean.append(cr)
        with open(args.output, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n  Results saved → {args.output}")


# =============================================================================
# PROGRAMMATIC API EXAMPLE
# =============================================================================
def demo_api():
    """Import and use the pipeline from another script."""
    pipeline = FracturePipeline(load_seg=True)

    result = pipeline.predict(
        image_path="xray.jpg",
        age=12,
        sex="F",
        location="wrist",
        save_mask=True,
    )

    if result["classification"]["fracture"]:
        grade  = result["segmentation"]["severity_grade"]
        weeks  = result["healing"]["healing_weeks"]
        ci_lo, ci_hi = result["healing"]["healing_range"]
        print(f"Fracture Grade {grade} — estimated healing: "
              f"{weeks} weeks  [{ci_lo}–{ci_hi}]")
    else:
        print("No fracture detected.")


if __name__ == "__main__":
    main()
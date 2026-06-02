"""
=============================================================================
INFERENCE PIPELINE v2 — Нэгтгэсэн хамгийн сайн загваруудын дамжлага
=============================================================================
Загварууд:
  1. mura_classifier.pt      — EfficientNet-B3, хугарал илрүүлэлт
  2. hybrid_unet_final.pth   — ResNet50 U-Net, сегментчлэл (Dice=0.604, TTA)
  3. Severity                — Маскийн morphological feature + heuristic
  4. grazped_healing.pt      — EfficientNet-B3 + meta, эдгэрэлт таамаглал

Хэрэглэх:
  python inference_v2.py --image path/to/xray.jpg
  python inference_v2.py --image xray.jpg --age 25 --sex M --location wrist
  python inference_v2.py --folder path/to/imgs/ --output results.json --save_mask
=============================================================================
"""

import os, sys, json, math, argparse, warnings
import pathlib
from pathlib import Path

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
import segmentation_models_pytorch as smp
from scipy import ndimage
from skimage import measure

warnings.filterwarnings("ignore")

# =============================================================================
# ЗАМУУД
# =============================================================================
_HERE = Path(__file__).resolve().parent

MURA_CKPT   = _HERE / "mura_classifier.pt"
HEAL_CKPT   = _HERE / "grazped_healing.pt"
# hybrid_unet_final.pth Turshilt_3/ дотор байна
HYBRID_CKPT = _HERE.parent / "Turshilt_3" / "hybrid_unet_final.pth"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_224  = 224   # MURA + Healing
IMG_384  = 384   # Hybrid U-Net

LOCATIONS = ["wrist", "forearm", "humerus", "tibia", "ankle", "clavicle", "femur"]

SEVERITY_DESC = {
    0: "Grade 0 — Хагарал байхгүй",
    1: "Grade 1 — Нарийн / hairline хагарал",
    2: "Grade 2 — Энгийн бүрэн хагарал",
    3: "Grade 3 — Нарийн хагарал / олон хэсэгт",
}

RECOVERY_TABLE = {
    0: (0,  0,  "Хагарал байхгүй"),
    1: (2,  4,  "Hairline — 2-4 долоо хоног"),
    2: (6,  8,  "Энгийн хагарал — 6-8 долоо хоног"),
    3: (10, 16, "Хүнд хагарал — 10-16 долоо хоног"),
}

# =============================================================================
# TRANSFORMS
# =============================================================================
TF_224 = A.Compose([
    A.Resize(IMG_224, IMG_224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

TF_384 = A.Compose([
    A.Resize(IMG_384, IMG_384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def load_tensor(path, size):
    tf = TF_224 if size == IMG_224 else TF_384
    img = np.array(Image.open(path).convert("RGB"))
    return tf(image=img)["image"].unsqueeze(0).to(DEVICE)


# =============================================================================
# 1. CLASSIFIER — EfficientNet-B3 (v1) OR ConvNeXt/EfficientNetV2 (v2)
# =============================================================================
class MURAClassifier(nn.Module):
    """v1 — EfficientNet-B3, 2-class softmax (mura_classifier.pt)"""
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


class FractureClassifierV2(nn.Module):
    """v2 — ConvNeXt / EfficientNetV2, binary sigmoid (fracture_classifier_v2.pt)"""
    def __init__(self, encoder_name="convnext_small.in22k", drop=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=False, num_classes=0, drop_rate=drop)
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(drop),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


def load_mura(path):
    """Checkpoint-ийн config-аас автоматаар v1/v2 таних."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg_saved = ckpt.get("config", {})
    encoder   = cfg_saved.get("encoder", "")
    img_size  = int(cfg_saved.get("img_size", IMG_224))

    if encoder and encoder != "efficientnet_b3":
        # v2 — ConvNeXt / EfficientNetV2
        m = FractureClassifierV2(encoder_name=encoder).to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        print(f"  [OK] Classifier v2 ({encoder[:24]}) ← {path.name}")
        return m, img_size, "v2"
    else:
        # v1 — EfficientNet-B3 (хуучин)
        m = MURAClassifier().to(DEVICE)
        m.load_state_dict(ckpt["model"])
        m.eval()
        print(f"  [OK] MURA classifier (EffB3)       ← {path.name}")
        return m, IMG_224, "v1"


# =============================================================================
# 2. HYBRID RESNET U-NET — сегментчлэл
# =============================================================================
def load_hybrid(path):
    m = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    # Checkpoint нь raw state_dict байна ({"model": ...} биш)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    m.load_state_dict(state)
    m.eval()
    print(f"  [OK] Hybrid ResNet U-Net   ← {path.name}  (Dice=0.604)")
    return m


@torch.inference_mode()
def predict_mask_tta(model, x):
    """4-fold TTA: оригинал + hflip + vflip + hvflip → дундаж probability"""
    probs = []
    for flip in [None, [-1], [-2], [-1, -2]]:
        xi = x if flip is None else torch.flip(x, dims=flip)
        pi = torch.sigmoid(model(xi))
        if flip is not None:
            pi = torch.flip(pi, dims=flip)
        probs.append(pi)
    return torch.stack(probs).mean(0)  # (1, 1, H, W)


# =============================================================================
# 3. SEVERITY — Morphological feature + heuristic
# =============================================================================
FEAT_COLS = [
    "area_px", "area_pct", "n_components", "largest_comp_pct",
    "major_axis", "minor_axis", "aspect_ratio", "eccentricity",
    "solidity", "bone_region",
]


def extract_morph_features(mask: np.ndarray) -> dict:
    """Binary маскаас 10 morphological feature гаргана."""
    h, w = mask.shape
    feats = dict(
        area_px=0.0, area_pct=0.0, n_components=0, largest_comp_pct=0.0,
        major_axis=0.0, minor_axis=0.0, aspect_ratio=0.0,
        eccentricity=0.0, solidity=0.0, bone_region=1,
    )
    if mask.sum() == 0:
        return feats

    feats["area_px"]  = float(mask.sum())
    feats["area_pct"] = float(mask.sum() / (h * w) * 100)

    labeled, n = ndimage.label(mask)
    feats["n_components"] = int(n)

    if n > 0:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        feats["largest_comp_pct"] = float(np.max(sizes) / mask.sum())

    props = measure.regionprops(labeled)
    if props:
        biggest = max(props, key=lambda p: p.area)
        feats["major_axis"]   = float(biggest.major_axis_length)
        feats["minor_axis"]   = float(biggest.minor_axis_length)
        feats["eccentricity"] = float(biggest.eccentricity)
        feats["solidity"]     = float(biggest.solidity)
        if biggest.minor_axis_length > 1e-6:
            feats["aspect_ratio"] = float(
                biggest.major_axis_length / biggest.minor_axis_length)
        cy, _ = biggest.centroid
        rel_y = cy / h
        feats["bone_region"] = 0 if rel_y < 0.33 else (2 if rel_y > 0.67 else 1)

    return feats


def heuristic_severity(feats: dict) -> int:
    """Morphological feature-аас severity grade (0-3) тодорхойлно."""
    area_px  = feats["area_px"]
    area_pct = feats["area_pct"]
    n_comp   = feats["n_components"]
    solidity = feats["solidity"]

    if area_px < 30:
        return 0
    if area_pct > 5.0 or n_comp >= 3 or (0 < solidity < 0.65):
        return 3
    if area_pct > 1.0 or n_comp == 2:
        return 2
    return 1


# =============================================================================
# 4. HEALING MODEL — EfficientNet-B3 + meta regression
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


class HealingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = timm.create_model(
            "efficientnet_b3", pretrained=False,
            features_only=True, out_indices=[4])
        self.heal_head = HealingHead(in_dim=384, meta_dim=4)

    def forward(self, x, meta):
        return self.heal_head(self.encoder(x)[0], meta)


def load_healing(path):
    m = HealingModel().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    print(f"  [OK] Healing regressor     ← {path.name}")
    return m


# =============================================================================
# INFERENCE АЛХМУУД
# =============================================================================
@torch.inference_mode()
def step1_classify(model, x, version="v1", threshold=0.5):
    logits = model(x)
    if version == "v2":
        # Binary sigmoid
        frac_prob = float(torch.sigmoid(logits.squeeze()).item())
        norm_prob  = 1.0 - frac_prob
        label      = int(frac_prob >= threshold)
    else:
        # 2-class softmax (v1)
        probs_list = F.softmax(logits, dim=1)[0].cpu().tolist()
        norm_prob, frac_prob = probs_list[0], probs_list[1]
        label = int(logits.argmax(1).item())
    return {
        "fracture":   bool(label == 1),
        "confidence": round(float(max(frac_prob, norm_prob)), 4),
        "probs": {
            "normal":   round(norm_prob, 4),
            "fracture": round(frac_prob, 4),
        },
    }


@torch.inference_mode()
def step2_segment(model, x):
    prob_t   = predict_mask_tta(model, x)           # (1,1,384,384)
    prob_np  = prob_t[0, 0].cpu().numpy()           # (384,384)
    mask     = (prob_np > 0.5).astype(np.uint8)
    coverage = float(mask.mean())
    return mask, prob_np, coverage


def step3_severity(mask: np.ndarray):
    feats    = extract_morph_features(mask)
    grade    = heuristic_severity(feats)
    rec_lo, rec_hi, rec_desc = RECOVERY_TABLE[grade]
    return {
        "grade":        grade,
        "label":        SEVERITY_DESC[grade],
        "mask_coverage": round(float(feats["area_pct"]), 4),
        "n_pieces":     int(feats["n_components"]),
        "solidity":     round(float(feats["solidity"]), 4),
        "recovery_weeks_min": rec_lo,
        "recovery_weeks_max": rec_hi,
        "recovery_desc": rec_desc,
    }


@torch.inference_mode()
def step4_healing(model, x, age, sex, location, severity):
    age_norm = age / 100.0
    sex_bin  = 1.0 if str(sex).upper() == "M" else 0.0
    loc_id   = (float(LOCATIONS.index(location)) / len(LOCATIONS)
                if location in LOCATIONS else 0.0)
    sev_norm = float(severity) / 3.0
    meta = torch.tensor(
        [[age_norm, sex_bin, loc_id, sev_norm]], dtype=torch.float32, device=DEVICE)
    pred, log_var = model(x, meta)
    weeks = float(pred.item())
    std   = float(torch.exp(0.5 * log_var).item())
    ci_lo = round(max(0.0, weeks - 1.96 * std), 1)
    ci_hi = round(weeks + 1.96 * std, 1)
    return {
        "weeks":         round(weeks, 1),
        "ci_95":         [ci_lo, ci_hi],
        "uncertainty_std": round(std, 2),
    }


# =============================================================================
# НЭГТГЭСЭН PIPELINE
# =============================================================================
class FracturePipelineV2:
    def __init__(self):
        print("\n" + "=" * 60)
        print("  FRACTURE PIPELINE v2 — загваруудыг ачааллаж байна")
        print("=" * 60)

        for ckpt, name in [(MURA_CKPT, "MURA"), (HYBRID_CKPT, "Hybrid U-Net"), (HEAL_CKPT, "Healing")]:
            if not ckpt.exists():
                print(f"  [WARN] {name} checkpoint олдсонгүй: {ckpt}")

        self.mura, self._cls_img_size, self._cls_ver = load_mura(MURA_CKPT)
        self.hybrid = load_hybrid(HYBRID_CKPT)
        self.heal   = load_healing(HEAL_CKPT)
        print("=" * 60 + "\n")

    def predict(self, image_path, age=25, sex="M", location="wrist",
                save_mask=False, force=False):
        """
        Нэг зурагт бүрэн pipeline ажиллуулна.

        Args:
            image_path : str | Path
            age        : int  — нас (жил)
            sex        : str  — 'M' эсвэл 'F'
            location   : str  — байршил (wrist, forearm, humerus, tibia, ankle, clavicle, femur)
            save_mask  : bool — маск PNG зургийн хажууд хадгалах
            force      : bool — ангилал "хагарал байхгүй" гарсан ч pipeline үргэлжлүүлэх

        Returns:
            dict
        """
        path  = Path(image_path)
        x_cls = load_tensor(path, self._cls_img_size)  # v1→224, v2→cfg.img_size
        x384  = load_tensor(path, IMG_384)

        result = {
            "image":   str(path),
            "patient": {"age": age, "sex": sex, "location": location},
        }

        # ── Алхам 1: Ангилал ──────────────────────────────────────────────
        cls = step1_classify(self.mura, x_cls, version=self._cls_ver)
        result["classification"] = cls

        if not cls["fracture"] and not force:
            result["classification"]["note"] = "Хагарал илрээгүй — pipeline зогслоо."
            result["segmentation"] = None
            result["severity"]     = None
            result["healing"]      = None
            return result
        if not cls["fracture"] and force:
            result["classification"]["note"] = "--force: pipeline үргэлжлэв."

        # ── Алхам 2: Сегментчлэл (Hybrid U-Net + TTA) ────────────────────
        mask, prob_np, coverage = step2_segment(self.hybrid, x384)

        if save_mask:
            mask_path = path.with_name(path.stem + "_mask.png")
            Image.fromarray(mask * 255).save(mask_path)

        result["segmentation"] = {
            "mask_coverage_pct": round(coverage * 100, 2),
            "mask_saved": str(mask_path) if save_mask else None,
        }

        # ── Алхам 3: Severity (morphological heuristic) ──────────────────
        sev = step3_severity(mask)
        result["severity"] = sev

        # ── Алхам 4: Эдгэрэлтийн таамаглал ──────────────────────────────
        x_heal = load_tensor(path, IMG_224)
        heal = step4_healing(
            self.heal, x_heal,
            age=age, sex=sex,
            location=location,
            severity=sev["grade"],
        )
        result["healing"] = heal

        return result

    def predict_batch(self, image_paths, age=25, sex="M", location="wrist", force=False):
        results = []
        for i, p in enumerate(image_paths, 1):
            print(f"  [{i}/{len(image_paths)}] {Path(p).name}")
            try:
                r = self.predict(p, age=age, sex=sex, location=location, force=force)
            except Exception as e:
                r = {"image": str(p), "error": str(e)}
            results.append(r)
        return results


# =============================================================================
# ХЭВЛЭХ
# =============================================================================
def print_result(r):
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Зураг   : {Path(r['image']).name}")

    if "error" in r:
        print(f"  АЛДАА   : {r['error']}")
        return

    pt = r.get("patient", {})
    print(f"  Өвчтөн  : нас={pt.get('age')}  хүйс={pt.get('sex')}  "
          f"байршил={pt.get('location')}")

    cls = r.get("classification", {})
    status = "ХАГАРАЛ ИЛЭРЛЭЭ" if cls.get("fracture") else "Хагарал байхгүй"
    print(f"\n  [1] Ангилал     : {status}  (итгэл={cls.get('confidence', 0):.1%})")
    print(f"      Магадлал    : хэвийн={cls['probs']['normal']:.3f}  "
          f"хагарал={cls['probs']['fracture']:.3f}")

    if not cls.get("fracture") and r.get("segmentation") is None:
        print(f"{sep}")
        return

    seg = r.get("segmentation", {}) or {}
    sev = r.get("severity", {}) or {}
    if sev:
        print(f"\n  [2] Сегментчлэл : маскийн хамрах хүрээ = "
              f"{seg.get('mask_coverage_pct', 0):.2f}%")
        print(f"\n  [3] Severity    : {sev.get('label', '')}")
        print(f"      Хэсгийн тоо : {sev.get('n_pieces', 0)}")
        print(f"      Solidity    : {sev.get('solidity', 0):.3f}")
        print(f"      Эдгэрэлт    : {sev.get('recovery_weeks_min')}-"
              f"{sev.get('recovery_weeks_max')} долоо хоног "
              f"({sev.get('recovery_desc', '')})")

    heal = r.get("healing", {}) or {}
    if heal:
        lo, hi = heal["ci_95"]
        print(f"\n  [4] Загварын    : {heal['weeks']} долоо хоног  "
              f"(95% CI: {lo}–{hi} дх,  σ={heal['uncertainty_std']})")

    print(sep)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Fracture X-ray — inference pipeline v2")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image",  type=str, help="Нэг рентген зурагны зам")
    grp.add_argument("--folder", type=str, help="Зургийн folder")
    p.add_argument("--age",      type=int,   default=25)
    p.add_argument("--sex",      type=str,   default="M", choices=["M", "F"])
    p.add_argument("--location", type=str,   default="wrist", choices=LOCATIONS)
    p.add_argument("--output",   type=str,   default="",
                   help="JSON үр дүнг хадгалах зам")
    p.add_argument("--save_mask", action="store_true",
                   help="Маск PNG зургийн хажуур хадгалах")
    p.add_argument("--force", action="store_true",
                   help="Ангилал 'хагарал байхгүй' гарсан ч дараагийн алхмуудыг ажиллуул")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = FracturePipelineV2()

    if args.image:
        result = pipeline.predict(
            args.image, age=args.age, sex=args.sex,
            location=args.location, save_mask=args.save_mask, force=args.force)
        print_result(result)
        results = [result]
    else:
        exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        paths = [p for p in Path(args.folder).iterdir()
                 if p.suffix.lower() in exts]
        print(f"  {len(paths)} зураг олдлоо: {args.folder}")
        results = pipeline.predict_batch(
            paths, age=args.age, sex=args.sex,
            location=args.location, force=args.force)
        for r in results:
            print_result(r)

    if args.output:
        clean = [{k: v for k, v in r.items() if k != "mask"} for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        print(f"\n  Үр дүн хадгалагдлаа → {args.output}")


if __name__ == "__main__":
    main()

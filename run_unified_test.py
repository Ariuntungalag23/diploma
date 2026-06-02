"""
574 unified test set дээр classification + segmentation inference.

Хэрэглэх:
    python run_unified_test.py              # FAST_MODE (1 fold, no TTA) — хурдан
    python run_unified_test.py --full       # 3 fold ensemble + 6-view TTA — нарийвчлалтай

Гарах:
    T3_classification/unified_test_predictions.csv
    T3_classification/unified_test_summary.json
"""

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import segmentation_models_pytorch as smp

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LAST_ROOT = Path('/Users/ariuntungalag/Desktop/LAST')
TEST_CSV  = LAST_ROOT / 'T3_classification' / 'data_unified_test.csv'

CLS_CKPT_DIR = LAST_ROOT / 'T3_classification' / 'checkpoints'
SEG_CKPT     = LAST_ROOT / 'Turshilt_3' / 'hybrid_unet_final.pth'
CV_SUMMARY   = CLS_CKPT_DIR / 'cv_summary.json'

CLS_MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384'
CLS_IMG_SIZE   = 384
SEG_ENCODER    = 'resnet50'
SEG_IMG_SIZE   = 384

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}')


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
class FractureModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            CLS_MODEL_NAME, pretrained=False,
            num_classes=0, global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def load_cls_models(folds=(0, 1, 2)):
    """Fold-уудыг ачаалах. Олдсонг нь буцаана."""
    models = []
    for f in folds:
        p = CLS_CKPT_DIR / f'fold{f}.pth'
        if not p.exists():
            continue
        m = FractureModel().to(DEVICE).eval()
        sd = torch.load(p, map_location=DEVICE, weights_only=False)
        if isinstance(sd, dict):
            state = sd.get('ema_state_dict') or sd.get('model_state_dict') \
                    or sd.get('state_dict') or sd
        else:
            state = sd
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
        m.load_state_dict(state, strict=False)
        models.append(m)
        print(f'  ✓ Loaded fold{f}.pth')
    return models


def build_cls_tta(size, hflip=False):
    tfs = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0),
    ]
    if hflip:
        tfs.append(A.HorizontalFlip(p=1.0))
    tfs.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return A.Compose(tfs)


@torch.no_grad()
def classify_image(img_rgb, models, tta_scales=(384,), hflip=False):
    """Олон fold × олон view → дундаж probability."""
    views = [(s, False) for s in tta_scales]
    if hflip:
        views += [(s, True) for s in tta_scales]
    probs = []
    for scale, flip in views:
        tf = build_cls_tta(scale, hflip=flip)
        x = tf(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
        for m in models:
            logit = m(x)
            probs.append(torch.sigmoid(logit.float()).cpu().item())
    return float(np.mean(probs))


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_seg_model():
    m = smp.Unet(encoder_name=SEG_ENCODER, encoder_weights=None,
                 in_channels=3, classes=1).to(DEVICE).eval()
    sd = torch.load(SEG_CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict):
        state = sd.get('model_state_dict') or sd.get('state_dict') or sd
    else:
        state = sd
    state = {k.replace('module.', ''): v for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    print('  ✓ Loaded segmentation U-Net')
    return m


def seg_preprocess(img_rgb_uint8, size=SEG_IMG_SIZE):
    pil = Image.fromarray(img_rgb_uint8).resize((size, size), Image.BILINEAR)
    arr = (np.array(pil, dtype=np.float32) / 255.0
           - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    arr = np.transpose(arr.astype(np.float32), (2, 0, 1))[None]
    return torch.from_numpy(arr).to(DEVICE)


@torch.no_grad()
def predict_mask(seg_model, img_rgb, use_tta=False, threshold=0.5):
    h, w = img_rgb.shape[:2]
    x = seg_preprocess(img_rgb)
    probs = [torch.sigmoid(seg_model(x))]

    if use_tta:
        # hflip
        xh = torch.flip(x, dims=[-1])
        ph = torch.sigmoid(seg_model(xh))
        probs.append(torch.flip(ph, dims=[-1]))
        # ±5° rotations
        for angle in (5, -5):
            theta = np.deg2rad(angle); cs, sn = np.cos(theta), np.sin(theta)
            rot = torch.tensor([[cs, -sn, 0], [sn, cs, 0]],
                               dtype=torch.float32, device=DEVICE).unsqueeze(0)
            grid = F.affine_grid(rot, x.shape, align_corners=False)
            xr = F.grid_sample(x, grid, align_corners=False)
            pr = torch.sigmoid(seg_model(xr))
            rot_inv = torch.tensor([[cs, sn, 0], [-sn, cs, 0]],
                                   dtype=torch.float32, device=DEVICE).unsqueeze(0)
            grid_inv = F.affine_grid(rot_inv, pr.shape, align_corners=False)
            probs.append(F.grid_sample(pr, grid_inv, align_corners=False))

    prob_avg = torch.stack(probs).mean(0)[0, 0].cpu().numpy()
    prob_full = cv2.resize(prob_avg, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob_full > threshold).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        nz = img[img > 0]
        if len(nz) > 100:
            lo, hi = np.percentile(nz, [1, 99])
        else:
            lo, hi = img.min(), img.max()
        hi = max(hi, lo + 1)
        img = np.clip(((img - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_gt_mask(ann_path, h, w):
    if not ann_path or not Path(ann_path).exists():
        return np.zeros((h, w), dtype=np.uint8)
    d = json.load(open(ann_path))
    mask = np.zeros((h, w), dtype=np.uint8)
    for o in d.get('objects', []):
        if o.get('classTitle') != 'fractured':
            continue
        g = o.get('geometryType')
        pts = o.get('points', {}).get('exterior', [])
        if g == 'polygon' and len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
        elif g == 'rectangle' and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    return mask


def dice_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    pg = p.sum() + g.sum()
    union = (p | g).sum()
    dice = float(2 * inter / max(pg, 1))
    iou = float(inter / max(union, 1))
    return dice, iou


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(full=False):
    print(f'\n{"="*60}')
    print(f'UNIFIED TEST SET INFERENCE — {"FULL" if full else "FAST"} mode')
    print(f'{"="*60}')

    # ─── Test CSV ─────────────────────────────────────────────────────────
    rows = list(csv.DictReader(open(TEST_CSV)))
    print(f'Test зураг: {len(rows)} (pos: {sum(1 for r in rows if r["label"]=="1")})')

    # ─── Calibrated threshold ─────────────────────────────────────────────
    threshold = 0.5
    if CV_SUMMARY.exists():
        threshold = json.load(open(CV_SUMMARY))['threshold']
        print(f'CV-аас threshold: {threshold:.4f}')

    # ─── Load models ──────────────────────────────────────────────────────
    print('\nCLS моделиуд ачааллаж байна...')
    folds = (0, 1, 2) if full else (0,)
    tta_scales = (320, 384, 448) if full else (384,)
    cls_models = load_cls_models(folds=folds)
    if not cls_models:
        print('  ! cls model олдсонгүй'); return

    print('SEG моделил ачааллаж байна...')
    seg_model = load_seg_model()

    # ─── Inference loop ───────────────────────────────────────────────────
    print(f'\n{len(rows)} зураг дээр inference...')
    print(f'  TTA scales: {tta_scales}, hflip: {full}, folds: {len(cls_models)}, seg_tta: {full}')
    t0 = time.time()

    results = []
    for i, r in enumerate(rows):
        try:
            img = read_image(r['image_path'])
        except Exception as e:
            print(f'  ! {r["image_path"]}: {e}')
            continue

        # Classification
        prob = classify_image(img, cls_models,
                              tta_scales=tta_scales, hflip=full)
        pred_cls = int(prob >= threshold)
        label = int(r['label'])

        out = {
            'image_path': r['image_path'],
            'label': label,
            'cls_prob': prob,
            'cls_pred': pred_cls,
        }

        # Segmentation — БОДИТ PIPELINE логик:
        # Зөвхөн ангилал нь "хагарал" (pred_cls == 1) бол л segmentation ажиллана.
        # Эс бөгөөс mask=None — хагарал илрээгүй гэж тооцно.
        if pred_cls == 1:
            pred_mask = predict_mask(seg_model, img, use_tta=full)
            gt_mask = load_gt_mask(r['ann_path'], *img.shape[:2])
            dice, iou = dice_iou(pred_mask, gt_mask)
            out.update({
                'seg_ran': True,
                'seg_dice': dice,
                'seg_iou': iou,
                'gt_has_polygon': bool(gt_mask.sum() > 0),
            })
        else:
            out['seg_ran'] = False

        results.append(out)

        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rows) - i - 1)
            print(f'  [{i+1:>3}/{len(rows)}]  elapsed {elapsed:.0f}s  eta {eta:.0f}s')

    print(f'\n  Inference done in {time.time()-t0:.0f} sec')

    # ─── Metrics ──────────────────────────────────────────────────────────
    y_true = np.array([r['label'] for r in results])
    y_prob = np.array([r['cls_prob'] for r in results])
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f'\n{"="*60}')
    print(f'CLASSIFICATION (n={len(y_true)})')
    print(f'{"="*60}')
    print(f'  AUC      : {auc:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Threshold: {threshold:.4f}')
    print(f'  Confusion Matrix [[TN, FP], [FN, TP]]: {cm}')
    print()
    print(classification_report(y_true, y_pred,
          target_names=['Not Fractured', 'Fractured'], digits=4))

    # ─── Segmentation үр дүн — Pipeline логиктой нийцсэн ─────────────────
    # Зөвхөн cls. нь "хагарал" гэж шийдсэн зургуудад segmentation ажилласан.
    #
    # TP (label=1, pred=1) → segm. ажилласан, polygon GT-той → Dice/IoU тооцох
    # FP (label=0, pred=1) → segm. ажилласан, GT хоосон mask → Dice=0 (зөв = байх ёсгүй)
    # FN (label=1, pred=0) → segm. алгассан → seg evaluation байхгүй
    # TN (label=0, pred=0) → segm. алгассан → зөв
    seg_ran = [r for r in results if r.get('seg_ran')]
    seg_tp  = [r for r in seg_ran if r['label'] == 1]    # бодит хагарал, дисплей хагарал
    seg_fp  = [r for r in seg_ran if r['label'] == 0]    # эрүүл, ангилал хагарал

    print(f'\n{"="*60}')
    print(f'SEGMENTATION (Pipeline-conditional)')
    print(f'{"="*60}')
    print(f'  Bvх классификаци: {len(results)} зураг')
    print(f'  Segm. ажилласан (cls=1): {len(seg_ran)} зураг')
    print(f'    └─ TP (бодит хагарал): {len(seg_tp)}')
    print(f'    └─ FP (хуурамч эерэг): {len(seg_fp)}')
    print(f'  Segm. алгасагдсан (cls=0): {len(results) - len(seg_ran)} зураг')

    seg_summary = {'n_ran': len(seg_ran),
                   'n_tp': len(seg_tp), 'n_fp': len(seg_fp)}

    if seg_tp:
        dice_tp = np.array([r['seg_dice'] for r in seg_tp])
        iou_tp  = np.array([r['seg_iou']  for r in seg_tp])
        seg_summary['tp_metrics'] = {
            'dice_mean': float(dice_tp.mean()),
            'dice_std':  float(dice_tp.std()),
            'dice_median': float(np.median(dice_tp)),
            'iou_mean':  float(iou_tp.mean()),
            'iou_std':   float(iou_tp.std()),
            'iou_median': float(np.median(iou_tp)),
        }
        print(f'\n  TP (n={len(seg_tp)}) — segm. үнэлгээ:')
        print(f'    Dice mean: {seg_summary["tp_metrics"]["dice_mean"]:.4f} ± {seg_summary["tp_metrics"]["dice_std"]:.4f}')
        print(f'    Dice median: {seg_summary["tp_metrics"]["dice_median"]:.4f}')
        print(f'    IoU  mean: {seg_summary["tp_metrics"]["iou_mean"]:.4f} ± {seg_summary["tp_metrics"]["iou_std"]:.4f}')
        print(f'    IoU  median: {seg_summary["tp_metrics"]["iou_median"]:.4f}')

    if seg_fp:
        # FP-д GT empty, pred mask non-empty → "хуурамч хагарал илрүүлсэн"
        fp_predicted_area = np.array([
            np.array([r for r in seg_fp if 'seg_dice' in r])
        ])
        n_fp_with_mask = sum(1 for r in seg_fp if r.get('seg_dice', 0) == 0
                             and r.get('seg_iou', 0) == 0)
        print(f'\n  FP (n={len(seg_fp)}) — хуурамч pred mask үүсгэсэн нь хор хөнөөл')

    # ─── Save ─────────────────────────────────────────────────────────────
    out_csv = LAST_ROOT / 'T3_classification' / 'unified_test_predictions.csv'
    with open(out_csv, 'w', newline='') as f:
        keys = ['image_path', 'label', 'cls_prob', 'cls_pred',
                'seg_ran', 'seg_dice', 'seg_iou', 'gt_has_polygon']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in keys})

    summary = {
        'mode': 'full' if full else 'fast',
        'n_test': len(y_true),
        'classification': {
            'auc': auc, 'f1': f1, 'accuracy': acc,
            'threshold': threshold,
            'confusion_matrix': cm,
        },
        'segmentation': seg_summary,
    }
    out_json = LAST_ROOT / 'T3_classification' / 'unified_test_summary.json'
    json.dump(summary, open(out_json, 'w'), indent=2)

    print(f'\n  ✓ {out_csv}')
    print(f'  ✓ {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true',
                        help='3-fold + 6-view TTA (slow, нарийвчлалтай)')
    args = parser.parse_args()
    main(full=args.full)

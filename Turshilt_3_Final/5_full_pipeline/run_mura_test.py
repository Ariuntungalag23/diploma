"""
MURA Stanford external classification test.

Дотоод сургалтад огт ороогүй Stanford MURA-ийн validation set дээр
ConvNeXt-Base 3-fold + 6-view TTA-аар үнэлгээ хийнэ.

Гарах файлууд:
  T3_classification/mura_predictions.csv
  T3_classification/mura_summary.json
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
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Config
LAST_ROOT = Path('/Users/ariuntungalag/Desktop/LAST')
MURA_ROOT = LAST_ROOT / 'MURA-v1.1'
VALID_CSV = MURA_ROOT / 'valid_image_paths.csv'

CLS_CKPT_DIR = LAST_ROOT / 'T3_classification' / 'checkpoints'
CV_SUMMARY = CLS_CKPT_DIR / 'cv_summary.json'

CLS_MODEL_NAME = 'convnext_base.fb_in22k_ft_in1k_384'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'Device: {DEVICE}')


class FractureModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            CLS_MODEL_NAME, pretrained=False, num_classes=0, global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def load_cls_models(folds=(0, 1, 2)):
    models = []
    for f in folds:
        p = CLS_CKPT_DIR / f'fold{f}.pth'
        if not p.exists():
            continue
        m = FractureModel().to(DEVICE).eval()
        sd = torch.load(p, map_location=DEVICE, weights_only=False)
        state = sd.get('ema_state_dict') or sd.get('model_state_dict') \
                or sd.get('state_dict') or sd
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
        m.load_state_dict(state, strict=False)
        models.append(m)
        print(f'  ✓ fold{f}.pth loaded')
    return models


def build_tta(size, hflip=False):
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


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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


@torch.no_grad()
def classify_image(img_rgb, models, tta_scales, hflip):
    views = [(s, False) for s in tta_scales]
    if hflip:
        views += [(s, True) for s in tta_scales]
    probs = []
    for scale, flip in views:
        tf = build_tta(scale, hflip=flip)
        x = tf(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
        for m in models:
            logit = m(x)
            probs.append(torch.sigmoid(logit.float()).cpu().item())
    return float(np.mean(probs))


def parse_label_and_part(rel_path):
    """MURA-v1.1/valid/XR_WRIST/patient11185/study1_positive/image1.png
       → ('XR_WRIST', 1) """
    parts = rel_path.split('/')
    body_part = parts[2] if len(parts) > 2 else 'UNKNOWN'
    label = 0
    for p in parts:
        if p.endswith('_positive'):
            label = 1; break
        if p.endswith('_negative'):
            label = 0; break
    return body_part, label


def main(full=False):
    print(f'\n{"="*60}')
    print(f'MURA EXTERNAL CLASSIFICATION TEST — {"FULL" if full else "FAST"}')
    print(f'{"="*60}')

    # Threshold from CV
    threshold = 0.2476
    if CV_SUMMARY.exists():
        threshold = json.load(open(CV_SUMMARY))['threshold']
    print(f'Threshold (CV-аас): {threshold:.4f}')

    # Load models
    folds = (0, 1, 2) if full else (0,)
    tta_scales = (320, 384, 448) if full else (384,)
    hflip = full
    print(f'\nFolds: {folds}, TTA scales: {tta_scales}, hflip: {hflip}')

    cls_models = load_cls_models(folds=folds)
    if not cls_models:
        print('! cls model олдсонгүй'); return

    # MURA valid CSV
    rel_paths = [line.strip() for line in open(VALID_CSV) if line.strip()]
    print(f'\nMURA validation: {len(rel_paths)} зураг')

    # Inference loop
    results = []
    t0 = time.time()
    for i, rel in enumerate(rel_paths):
        body_part, label = parse_label_and_part(rel)
        # rel starts with "MURA-v1.1/..." which is relative to ROOT parent
        img_path = LAST_ROOT / rel
        if not img_path.exists():
            continue

        try:
            img = read_image(img_path)
        except Exception as e:
            continue

        prob = classify_image(img, cls_models, tta_scales, hflip)
        pred = int(prob >= threshold)

        results.append({
            'image_path': str(img_path),
            'body_part': body_part,
            'label': label,
            'cls_prob': prob,
            'cls_pred': pred,
        })

        if (i + 1) % 100 == 0 or i == len(rel_paths) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rel_paths) - i - 1)
            print(f'  [{i+1:>4}/{len(rel_paths)}]  elapsed {elapsed:.0f}s  eta {eta:.0f}s')

    print(f'\n  Inference done in {time.time()-t0:.0f}s')

    # Metrics
    y_true = np.array([r['label'] for r in results])
    y_prob = np.array([r['cls_prob'] for r in results])
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f'\n{"="*60}')
    print(f'OVERALL (n={len(y_true)})')
    print(f'{"="*60}')
    print(f'  AUC      : {auc:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Threshold: {threshold:.4f}')
    print(f'  CM [[TN, FP], [FN, TP]]: {cm}')
    print()
    print(classification_report(y_true, y_pred,
          target_names=['Negative', 'Positive (Fracture)'], digits=4))

    # Per body part
    print(f'\n{"="*60}')
    print(f'PER BODY PART AUC')
    print(f'{"="*60}')
    per_part = {}
    for part in sorted(set(r['body_part'] for r in results)):
        idx = [i for i, r in enumerate(results) if r['body_part'] == part]
        if len(set(y_true[idx])) < 2:
            continue
        p_auc = float(roc_auc_score(y_true[idx], y_prob[idx]))
        p_f1 = float(f1_score(y_true[idx], y_pred[idx]))
        per_part[part] = {'n': len(idx), 'auc': p_auc, 'f1': p_f1,
                          'pos': int(y_true[idx].sum())}
        print(f'  {part:15s}  n={len(idx):4d}  pos={int(y_true[idx].sum()):4d}  '
              f'AUC={p_auc:.4f}  F1={p_f1:.4f}')

    # Save
    out_csv = LAST_ROOT / 'T3_classification' / 'mura_predictions.csv'
    with open(out_csv, 'w', newline='') as f:
        keys = ['image_path', 'body_part', 'label', 'cls_prob', 'cls_pred']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in results])

    out_json = LAST_ROOT / 'T3_classification' / 'mura_summary.json'
    summary = {
        'mode': 'full' if full else 'fast',
        'n_test': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
        'overall': {
            'auc': auc, 'f1': f1, 'accuracy': acc,
            'threshold': threshold, 'confusion_matrix': cm,
        },
        'per_body_part': per_part,
    }
    json.dump(summary, open(out_json, 'w'), indent=2)
    print(f'\n  ✓ {out_csv}')
    print(f'  ✓ {out_json}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    main(full=args.full)

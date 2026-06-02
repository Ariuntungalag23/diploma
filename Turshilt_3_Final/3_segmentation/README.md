# 3. Segmentation — Hybrid ResNet50 U-Net

Хагарлын бүсийн pixel-level segmentation model.

## Файлууд

| Файл | Тайлбар |
|---|---|
| `sam_auto_annotation.ipynb` | Шат A: SAM ViT-H-аар auto-annotation (4 dataset, ~3K pseudo-mask) |
| `hybrid_resnet_unet_training.ipynb` | Сургалтын notebook (3-шаттай: pretrain → refine → TTA) |
| `hybrid_unet_final.pth` | **Эцсийн checkpoint** (symlink, 130 MB) |

## Архитектур

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',   # ImageNet pretrained
    in_channels=3,
    classes=1,                     # binary mask + sigmoid
)
IMG_SIZE = 384
```

## Сургалт — 2 шатлал

### Шат A: SAM auto-mask pretrain (3K)

```
4 dataset (FracAtlas, YOLOBoneFracture, BoneFractureCVProject, BoneFracture4th)
   ↓
SAM ViT-H + bounding box prompt
   ↓ Filter: score ≥ 0.70, area 0.3%-30%, metal implant exclude
~3,000 pseudo-mask
   ↓
Pretrain (15 epoch, lr=1e-4, batch=16, patience=5)
```

### Шат B: FracAtlas жинхэнэ маск refine (574)

```
FracAtlas DatasetNinja train/ (574 хагарал + Supervisely polygon)
   ↓
Refine (40 epoch, lr=2e-5  ★ маш бага LR, batch=8, patience=10)
```

### Loss

```python
0.5 × smp.losses.DiceLoss(mode='binary')
+ 0.5 × nn.BCEWithLogitsLoss()
```

### Optimizer
```python
torch.optim.AdamW(lr=lr, weight_decay=1e-4)
+ CosineAnnealingLR
```

## Inference TTA (Шат C) — 4-view

```python
@torch.no_grad()
def predict_mask_tta(img, threshold=0.5):
    x = preprocess(img)                       # 384×384, ImageNet normalize
    probs = [
        sigmoid(model(x)),                    # 1) Original
        torch.flip(sigmoid(model(flip(x))), dims=[-1]),  # 2) Hflip
        # 3-4) +5° / -5° rotations (with inverse rotate)
    ]
    avg = mean(probs)
    return cv2.resize(avg, (orig_w, orig_h)) > threshold
```

## Үр дүн

| Туршилт | Test set | Dice mean | Dice median |
|---|---|---:|---:|
| Хуучин (DiffusionTransUNet) | FracAtlas test | 0.16 | — |
| Hybrid U-Net + TTA | FracAtlas test (61) | 0.42 | 0.53 |
| **Pipeline-conditional (TP=43)** | 574 unified test | **0.477** | **0.578** |

Литературын benchmark: Dice 0.55-0.75 — median утга нь дотроо ороод байгаа.

## Эх байршил

`/Users/ariuntungalag/Desktop/LAST/Turshilt_3/hybrid_unet_final.pth`

## Сургалтын notebook-уудын дотоод тохиргоо

- `sam_auto_annotation.ipynb`: SAM ViT-H, металл filter, CLAHE preprocessing
- `hybrid_resnet_unet_training.ipynb`: ImageNet pretrained, AdamW, CosineLR

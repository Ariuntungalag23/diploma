# Bone Fracture Classification — Pipeline

Pediatric + adult bone fracture binary classification model ConvNeXt-Base + 3-fold CV + TTA ensemble.

## 📦 Файлын бүтэц

```
fracture_classifier/
├── config.py            # Бүх hyperparameter, path
├── build_dataset.py     # 4 dataset → нэгдсэн CSV
├── data.py              # Dataset, 16-bit window/level, augmentation
├── model.py             # ConvNeXt-Base + EMA + LLD optimizer
├── train.py             # 3-fold CV сургалт
├── infer.py             # TTA + ensemble inference
└── download_kaggle.sh   # 4 dataset татах
```

## 🔧 Install

```bash
pip install torch torchvision timm albumentations opencv-python \
            pandas scikit-learn wandb kaggle kagglehub
```

## ⚙️ Хэрэглэх 4 алхам

### 1. Dataset татах (~6.5 GB)

Kaggle API token эхлээд бэлдэх:
```bash
# https://www.kaggle.com/settings → "Create New API Token" → kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Татах:
```bash
bash download_kaggle.sh
# → datasets/ хавтсанд 4 dataset бэлэн болно
```

### 2. CSV үүсгэх

```bash
python build_dataset.py --root ./datasets
# → data_train.csv (85%) + data_test.csv (15%) үүсгэнэ
```

Гаралт:
```
TRAIN/TEST SPLIT (Group-level):
  TRAIN (CV дотор): 28,500 зураг (4,200 pos, 14.7%)
    bonefracturecv      :  4,000 (1,200 pos)
    fracatlas           :  3,470 (610 pos)
    graz                : 17,278 (1,800 pos)
    multiregion         :  3,752 (590 pos)
  TEST (hold-out): 5,000 зураг (730 pos)
  Group overlap (заавал 0): 0
  Санал болгох pos_weight (BCE): 5.785
```

### 3. WandB login (анхны удаа)

```bash
pip install wandb
wandb login
# API key оруулна (https://wandb.ai/authorize)
```

### 4. Сургалт + Inference

```bash
# 3-fold CV сургалт (~9 цаг A6000 дээр)
python train.py

# Test set дээр TTA + ensemble (~30 минут)
python infer.py
```

## 📊 Хүлээгдэж буй үр дүн

| Metric | Утга |
|--------|------|
| CV OOF AUC | ~0.94-0.96 |
| Test AUC (hold-out) | ~0.93-0.95 |
| Recall (threshold-той) | ≥ 0.95 |
| Precision (recall=0.95 дээр) | ~0.65-0.75 |

## ⚙️ Тохиргоо өөрчлөх

`config.py` дотор бүх hyperparameter байгаа. Эсвэл command-line:

```bash
python train.py --epochs 20 --batch_size 24 --n_folds 5
python train.py --no-wandb  # WandB ашиглахгүй
```

## 🔍 Алдаа гарвал шалгах зүйлс

**1. `dataset.csv` олдсонгүй (GRAZPEDWRI)**
```bash
ls datasets/GRAZPEDWRI-DX/
# dataset.csv харагдах ёстой
```

**2. CUDA out of memory**
```bash
python train.py --batch_size 16  # batch_size бууруулах
```
Эсвэл `config.py`-д `cfg.accumulate = 4` болгоход effective batch хадгалагдана.

**3. Roboflow augmented хувилбарууд хэт олон бол**
```bash
python build_dataset.py --root ./datasets --include_roboflow_augmented
```

**4. WandB-гүй ажиллуулах**
```bash
python train.py --no-wandb
```

## 📝 Datasets

| # | Dataset | Source |
|---|---------|--------|
| 1 | GRAZPEDWRI-DX | [Kaggle: jasonroggy/grazpedwri-dx](https://www.kaggle.com/datasets/jasonroggy/grazpedwri-dx) |
| 2 | FracAtlas | [Kaggle: tommyngx/fracatlas](https://www.kaggle.com/datasets/tommyngx/fracatlas) |
| 3 | Bone Fracture Detection CV | [Kaggle: pkdarabi/...](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) |
| 4 | Multi-Region X-ray | [Kaggle: bmadushanirodrigo/...](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data) |

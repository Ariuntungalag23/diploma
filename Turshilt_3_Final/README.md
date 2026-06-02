# Турших 3 — Бүрэн материал

Дипломын ажлын Турших 3 хэсэгт хамаарах бүх материал.

## Pipeline тойм

```
[X-ray зураг]
    ↓
[1. Classification]   →  ConvNeXt-Base × 3-fold ensemble + 6-view TTA
    ↓                     (Хагарал байна уу? P ≥ 0.2476 бол тийм)
[2. Segmentation]     →  Hybrid ResNet50 U-Net + 4-view TTA  (Cls=1 үед л ажиллана)
    ↓                     (Хугарлын маск гаргах)
[3. Severity]         →  10 морфологийн feature → RandomForest (4 анги)
    ↓
[4. Healing time]     →  Lookup table + Gaussian (нас тохируулна)
    ↓
[6-panel dashboard]
```

## Folder бүтэц

| Folder | Агуулга |
|---|---|
| **1_data_preparation/** | Сургалт/тест өгөгдөл бэлдэх кодууд + CSV-ууд |
| **2_classification/** | ConvNeXt-Base ангилагч (сургалт + checkpoint + inference) |
| **3_segmentation/** | ResNet50 U-Net (SAM auto-mask + FracAtlas refine training) |
| **4_severity/** | RandomForest хүндрэлийн зэрэг ангилагч |
| **5_full_pipeline/** | End-to-end pipeline + тестийн скрипт + үр дүн |
| **6_training_reference/** | Хуучин туршилт сургалтын скриптүүд (lit. сонгох reference) |
| **7_documentation/** | Үндсэн docx тайлбар + scripted generator |

## Үндсэн үр дүн

### Internal hold-out (574 unified test)

| Шат | Метрик | Утга |
|---|---|---|
| Classification | Test AUC | **0.9703** (CV 0.9887, gap 0.018) |
| Classification | Test F1 | 0.8037 |
| Classification | Test Accuracy | 0.9634 |
| Segmentation (TP=43) | Dice mean | **0.4770 ± 0.3391** |
| Segmentation (TP=43) | Dice median | 0.5779 |

### MURA external (Stanford, 3,197 зураг — огт ороогүй)

| Метрик | Утга |
|---|---|
| AUC | **0.8323** |
| F1 | 0.7432 |
| Accuracy | 0.7376 |
| Best body parts | FOREARM (0.93), HUMERUS (0.93), WRIST (0.87) |

Pipeline логик: Зөвхөн classification нь "хагарал" гэж шийдсэн зургуудад
segmentation ажилладаг. 574 зурагнаас 46 (43 TP + 3 FP) дээр segm. явсан.

## Шат бүрд хэрэгтэй ачаалал

- **Classification**: `python 2_classification/infer.py` (3-fold + TTA)
- **Сегментаци + severity**: `5_full_pipeline/t3_full_pipeline.ipynb` notebook (Jupyter)
- **Бүх pipeline 574 test дээр**: `python 5_full_pipeline/run_unified_test.py --full`

## Шаардлагатай нөхцөл (зөвхөн inference)

```bash
pip install torch torchvision timm albumentations opencv-python \
            pandas scikit-learn scikit-image Pillow tqdm \
            segmentation-models-pytorch
```

Mac дээр (MPS): pytorch 2.0+ автомат MPS-ийг таних
CUDA дээр: pytorch + cudatoolkit
CPU дээр: ажиллах боловч хурд буурна

## Дэлгэрэнгүй тайлбар

[7_documentation/Turshilt3_Tailbar.docx](7_documentation/Turshilt3_Tailbar.docx)
дотроос бүх шатанд хийсэн ажил, тестийн методологи, хязгаарлалт, үр дүнг харна.

## Файлын лицейн ялгаа

- **Кодын файлууд**: txt/py/ipynb — шууд хуулсан
- **Том model файлууд** (.pth, .pkl): эх байршил руу **symbolic link**
  үүсгэсэн. Disk space хэмнэх зорилгоор.

Эх байршлууд:
- `2_classification/checkpoints/fold[0-2].pth` → `../../../T3_classification/checkpoints/`
- `3_segmentation/hybrid_unet_final.pth` → `../../Turshilt_3/`
- `4_severity/severity_classifier.pkl` → `../../`

Хэрэв LAST/T3_classification/ эсвэл LAST/Turshilt_3/ устгасан бол link
ажиллахгүй — эх checkpoint-уудыг хадгал.

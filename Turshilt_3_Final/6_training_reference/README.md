# 6. Сургалтын код — Reference

Дипломын ажилд **шууд ашиглаагүй** боловч үндсэн судалгаанд оруулсан
сургалтын скриптүүд (history/reference).

## Файлууд

### Classifier v2 (одоо үндсэн classification model болсон)

| Файл | Тайлбар |
|---|---|
| `train_classifier_v2.py` | A6000 48GB-д зориулсан ConvNeXt-Small + 2 шаттай сургалт |
| `train_classifier_v2_colab.ipynb` | Colab хувилбар |
| `baseline_comparison.py` | ResNet50 vs EffNet-B3 vs ConvNeXt-S харьцуулалт |

> **Жич**: Дипломд эцсийн хувилбар нь ConvNeXt-**Base** болж сонгогдсон —
> 2_classification/ folder-т.

### Хуучин 3-stage approach (зөвхөн reference)

| Файл | Тайлбар |
|---|---|
| `train_1_mura.py` | EfficientNet-B3 дээр MURA dataset (хагарал/эрүүл) |
| `train_2_fracatlas.py` | SAM-LoRA segmentation + severity (EffNet encoder, FracAtlas) |
| `train_3_grazped.py` | GRAZPEDWRI дээр healing time regression (синтетик label) |

> **Жич**: Эдгээр нь дипломын ажлын **анхны хувилбарт** хэрэглэгдсэн боловч
> Turshilt 3-д ConvNeXt-Base + Hybrid U-Net + RF + lookup table-аар сольсон.

## Эх байршил

`/Users/ariuntungalag/Desktop/LAST/TRAINING_CODE/`

## Яагаад reference хадгалсан вэ?

1. **Архитектурын эволюци**-ийг тэмдэглэхэд (хуучин → шинэ)
2. Defense-д "яагаад тэр архитектурыг сонгосон" гэдэг асуултанд хариулах
3. Литературын харьцуулалтын нотолгоо

## requirements.txt

Эдгээр кодыг ажиллуулахад шаардлагатай pip packages.

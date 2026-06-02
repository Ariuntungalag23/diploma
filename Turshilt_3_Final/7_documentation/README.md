# 7. Документ + Generator

## Файлууд

| Файл | Тайлбар |
|---|---|
| `Turshilt3_Tailbar.docx` | **Үндсэн дипломын тайлбар** (~50 KB, 9 хэсэгтэй) |
| `make_turshilt3_doc.py` | Docx-ийг автоматаар үүсгэх Python script (python-docx ашиглана) |

## Docx-ийн агуулга

| § | Хэсэг |
|---|---|
| 1 | Ангилал (Classification) — өгөгдөл, архитектур, сургалт, TTA, threshold |
| 2 | Сегментаци (Segmentation) — 2 шатлал, ResNet50 U-Net, TTA |
| 3 | Severity — Heuristic шошго, 10 feature, RandomForest |
| 4 | Healing — Lookup table (сургалт байхгүй) |
| 5 | End-to-end pipeline |
| 6 | Нэгтгэсэн хүснэгт |
| 7 | Test set + үнэлгээ методологи |
| **8** | **574 unified test үр дүн** (Cls AUC 0.97, Seg Dice 0.48 TP-д) |
| 9 | Хязгаарлалтууд (ил тод бичсэн) |

## Docx-ийг дахин үүсгэх

```bash
pip install --user --break-system-packages python-docx

python make_turshilt3_doc.py
# → Turshilt3_Tailbar.docx (overwrite)
```

Скриптийг засаад ажиллуулахад docx тэр даруй шинэчлэгдэнэ.

#!/bin/bash
# ============================================================================
# Bone Fracture Classification — БҮХ 4 DATASET KAGGLE-ЭЭС ТАТАХ
# ============================================================================
# kaggle.json-ыг скрипттэй ИЖИЛ ХАВТАСТ тавьна (~/.kaggle/ биш!)
#
# Шаардлага:
#   1. ~20 GB сул диск
#   2. pip install kaggle
#   3. kaggle.json — скрипттэй ижил хавтаст:
#      → kaggle.com/settings → "Create New API Token" → kaggle.json татна
#      → энэ скрипттэй ИЖИЛ хавтаст хуулна
#
# Хэрэглэх:
#   bash download_kaggle.sh
#
# Эсвэл өөр зам зааж өгөх:
#   DATA_ROOT=/data/fracture bash download_kaggle.sh
# ============================================================================

set -e

# Скриптийн байрлах хавтас (kaggle.json энд байх ёстой)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KAGGLE_JSON="$SCRIPT_DIR/kaggle.json"

DATA_ROOT="${DATA_ROOT:-./datasets}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "================================================================"
echo "Татах хавтас: $(pwd)"
echo "kaggle.json:  $KAGGLE_JSON"
echo "================================================================"

# ----------------------------------------------------------------------------
# kaggle.json шалгалт
# ----------------------------------------------------------------------------
if [ ! -f "$KAGGLE_JSON" ]; then
    echo ""
    echo "❌ kaggle.json олдсонгүй: $KAGGLE_JSON"
    echo ""
    echo "Алхамууд:"
    echo "  1. https://www.kaggle.com/settings руу ор"
    echo "  2. 'Create New API Token' дарж kaggle.json татна"
    echo "  3. Татсан kaggle.json-ыг ЭНЭ хавтаст хуулна:"
    echo "       $SCRIPT_DIR/"
    echo "  4. дахин ажиллуулна: bash download_kaggle.sh"
    exit 1
fi

# Kaggle CLI-д credentials байршлыг заах
# (~/.kaggle/ руу copy хийх шаардлагагүй)
export KAGGLE_CONFIG_DIR="$SCRIPT_DIR"

# kaggle.json-н permission 600 байх ёстой (Kaggle CLI шаарддаг)
chmod 600 "$KAGGLE_JSON"

# Kaggle package шалгалт
if ! command -v kaggle &> /dev/null; then
    echo "  → kaggle package суулгаж байна..."
    pip install kaggle --quiet
fi

# ----------------------------------------------------------------------------
# 1) GRAZPEDWRI-DX — олон давхар zip-тэй учир нэмж задлах хэрэгтэй
# ----------------------------------------------------------------------------
echo ""
echo ">>> [1/4] GRAZPEDWRI-DX (Kaggle: jasonroggy/grazpedwri-dx, ~16 GB)"
echo ""

if [ ! -d "GRAZPEDWRI-DX" ] || [ -z "$(find GRAZPEDWRI-DX -name '*.png' -o -name '*.jpg' 2>/dev/null | head -1)" ]; then
    mkdir -p GRAZPEDWRI-DX && cd GRAZPEDWRI-DX

    # 1.1) Гадаад zip-ийг татаад задлах
    echo "  → outer zip татаж байна (~16 GB)..."
    kaggle datasets download -d jasonroggy/grazpedwri-dx --unzip

    # 1.2) Дотор нь үлдсэн images_partN.zip бүгдийг задлах
    echo "  → inner zip-үүдийг задалж байна..."
    for z in images_part*.zip; do
        if [ -f "$z" ]; then
            echo "    • $z задалж байна..."
            unzip -q -o "$z"
            rm -f "$z"        # дискний зайг чөлөөлөх (заавал биш)
        fi
    done

    cd ..
    echo "  ✅ GRAZPEDWRI-DX бэлэн"
else
    echo "  ⚠ GRAZPEDWRI-DX/ хавтас аль хэдийн байна, алгаслаа"
fi

# ----------------------------------------------------------------------------
# 2) FracAtlas — Kaggle mirror (323 MB)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [2/4] FracAtlas (Kaggle: tommyngx/fracatlas, ~323 MB)"
echo ""

if [ ! -d "FracAtlas" ]; then
    mkdir -p FracAtlas && cd FracAtlas
    kaggle datasets download -d tommyngx/fracatlas --unzip
    cd ..
    echo "  ✅ FracAtlas бэлэн"
else
    echo "  ⚠ FracAtlas/ хавтас аль хэдийн байна, алгаслаа"
fi

# ----------------------------------------------------------------------------
# 3) Bone Fracture Detection CV (Kaggle: pkdarabi, 4,148 зураг)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [3/4] Bone Fracture Detection CV (Kaggle: pkdarabi/..., ~500 MB)"
echo ""

if [ ! -d "BoneFractureCV" ]; then
    mkdir -p BoneFractureCV && cd BoneFractureCV
    kaggle datasets download -d pkdarabi/bone-fracture-detection-computer-vision-project --unzip
    cd ..
    echo "  ✅ BoneFractureCV бэлэн"
else
    echo "  ⚠ BoneFractureCV/ хавтас аль хэдийн байна, алгаслаа"
fi

# ----------------------------------------------------------------------------
# 4) Bone Fracture Multi-Region X-ray (10,580 зураг!)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [4/4] Bone Fracture Multi-Region (Kaggle: bmadushanirodrigo/..., ~600 MB)"
echo ""

if [ ! -d "MultiRegionXray" ]; then
    mkdir -p MultiRegionXray && cd MultiRegionXray
    kaggle datasets download -d bmadushanirodrigo/fracture-multi-region-x-ray-data --unzip
    cd ..
    echo "  ✅ MultiRegionXray бэлэн"
else
    echo "  ⚠ MultiRegionXray/ хавтас аль хэдийн байна, алгаслаа"
fi

# ----------------------------------------------------------------------------
# Дүгнэлт
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "✅ БҮГДИЙГ АМЖИЛТТАЙ ТАТСАН"
echo "================================================================"
echo ""
echo "Хавтсын хэмжээ:"
du -sh "$DATA_ROOT"/*/ 2>/dev/null
echo ""
echo "Зургийн тоо:"
find "$DATA_ROOT/GRAZPEDWRI-DX"   \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | xargs echo "  GRAZPEDWRI-DX  :"
find "$DATA_ROOT/FracAtlas"        \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | xargs echo "  FracAtlas      :"
find "$DATA_ROOT/BoneFractureCV"   \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | xargs echo "  BoneFractureCV :"
find "$DATA_ROOT/MultiRegionXray"  \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l | xargs echo "  MultiRegionXray:"
echo ""
echo "Дараагийн алхам:"
echo "  python build_dataset.py --root $DATA_ROOT"

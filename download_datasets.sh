#!/bin/bash
# Кагглээс 3 dataset татах (FracAtlas аль хэдийн локалд бий)
# Run: bash download_datasets.sh

set -e

LAST_ROOT="/Users/ariuntungalag/Desktop/LAST"
DATA_ROOT="$LAST_ROOT/datasets"
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "================================================================"
echo "  Татах хавтас: $DATA_ROOT"
echo "  kaggle.json:   $KAGGLE_JSON"
echo "================================================================"

# Permission шалгах
chmod 600 "$KAGGLE_JSON"

# kaggle CLI кодыг шалгах
export PATH="$HOME/Library/Python/3.13/bin:$PATH"
if ! command -v kaggle &> /dev/null; then
    echo "❌ kaggle CLI олдсонгүй. python3 -m pip install --user kaggle ажиллуулна уу."
    exit 1
fi

# ----------------------------------------------------------------------------
# 1) GRAZPEDWRI-DX (~16 GB)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [1/3] GRAZPEDWRI-DX (jasonroggy/grazpedwri-dx, ~16 GB)"

if [ ! -d "GRAZPEDWRI-DX" ] || [ -z "$(find GRAZPEDWRI-DX -name '*.png' 2>/dev/null | head -1)" ]; then
    mkdir -p GRAZPEDWRI-DX && cd GRAZPEDWRI-DX
    echo "  → Outer zip татаж задлаж байна..."
    kaggle datasets download -d jasonroggy/grazpedwri-dx --unzip

    echo "  → Inner zip-үүдийг задалж байна..."
    for z in images_part*.zip; do
        if [ -f "$z" ]; then
            echo "    • $z"
            unzip -q -o "$z"
            rm -f "$z"
        fi
    done
    cd ..
    echo "  ✅ GRAZPEDWRI-DX бэлэн"
else
    echo "  ⚠ Аль хэдийн бий"
fi

# ----------------------------------------------------------------------------
# 2) BoneFractureCV (~500 MB)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [2/3] BoneFractureCV (pkdarabi/...)"

if [ ! -d "BoneFractureCV" ]; then
    mkdir -p BoneFractureCV && cd BoneFractureCV
    kaggle datasets download -d pkdarabi/bone-fracture-detection-computer-vision-project --unzip
    cd ..
    echo "  ✅ BoneFractureCV бэлэн"
else
    echo "  ⚠ Аль хэдийн бий"
fi

# ----------------------------------------------------------------------------
# 3) Multi-Region X-ray (~600 MB)
# ----------------------------------------------------------------------------
echo ""
echo ">>> [3/3] MultiRegionXray (bmadushanirodrigo/...)"

if [ ! -d "MultiRegionXray" ]; then
    mkdir -p MultiRegionXray && cd MultiRegionXray
    kaggle datasets download -d bmadushanirodrigo/fracture-multi-region-x-ray-data --unzip
    cd ..
    echo "  ✅ MultiRegionXray бэлэн"
else
    echo "  ⚠ Аль хэдийн бий"
fi

# ----------------------------------------------------------------------------
# Дүгнэлт
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Татаж дууссан"
echo "================================================================"
du -sh "$DATA_ROOT"/*/

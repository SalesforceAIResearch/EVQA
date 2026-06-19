#!/bin/bash
# Layered upload strategy for st-evidence-instruct dataset
# Upload metadata first, then optionally upload compressed media files

set -e

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="shijiewang/st-evidence-instruct"

echo "============================================================"
echo "ST-Evidence-Instruct Layered Upload to Hugging Face"
echo "============================================================"

# Step 1: Upload metadata only (CSV files, README, etc)
echo ""
echo "Step 1: Uploading metadata (CSV, README, annotations)..."
echo "------------------------------------------------------------"

TEMP_METADATA="/tmp/st-evidence-metadata-$$"
mkdir -p "$TEMP_METADATA"

# Copy only metadata files
rsync -av --progress \
  --include='*.csv' \
  --include='*.md' \
  --include='*.json' \
  --include='*.pkl' \
  --include='*/' \
  --exclude='*' \
  "$SOURCE_DIR/" "$TEMP_METADATA/"

echo "Metadata size:"
du -sh "$TEMP_METADATA"

huggingface-cli upload "$REPO_ID" "$TEMP_METADATA" --repo-type dataset

rm -rf "$TEMP_METADATA"
echo "✓ Metadata uploaded"

# Step 2: Ask about media files
echo ""
echo "============================================================"
echo "Step 2: Media files (video_frames and masks)"
echo "============================================================"
echo ""
echo "You have two options for media files:"
echo ""
echo "A) Upload as compressed archives (faster, ~GB level)"
echo "   - Pros: Fast upload, single files"
echo "   - Cons: Users need to extract before use"
echo ""
echo "B) Upload as individual files (slower, better UX)"
echo "   - Pros: Direct access, HF datasets compatible"
echo "   - Cons: Slower upload, many files"
echo ""
read -p "Choose option (A/B/skip): " choice

case $choice in
  [Aa]*)
    echo ""
    echo "Creating compressed archives..."

    # Compress vicas data
    echo "Compressing vicas video_frames and masks..."
    TEMP_VICAS="/tmp/vicas-media-$$"
    mkdir -p "$TEMP_VICAS/gen_qa/vicas"

    tar -czf "$TEMP_VICAS/gen_qa/vicas/video_frames.tar.gz" \
      -C /fsx/home/shijie.wang/code/UniPixel/data/vicas video_frames

    tar -czf "$TEMP_VICAS/gen_qa/vicas/masks.tar.gz" \
      -C /fsx/home/shijie.wang/code/UniPixel/data/vicas masks

    echo "✓ ViCaS archives created"

    # Compress st-evidence data
    echo "Compressing st-evidence video_frames and masks..."
    mkdir -p "$TEMP_VICAS/gen_mask"

    tar -czf "$TEMP_VICAS/gen_mask/video_frames_6fps.tar.gz" \
      -C /fsx/home/shijie.wang/code/st-evidence/training_annotations video_frames_6fps

    tar -czf "$TEMP_VICAS/gen_mask/masks.tar.gz" \
      -C /fsx/home/shijie.wang/code/st-evidence/training_annotations/mask_results/shared masks_all_clevrer_gdino_refined

    echo "✓ ST-Evidence archives created"

    # Show sizes
    echo ""
    echo "Archive sizes:"
    du -sh "$TEMP_VICAS"/*/*/*

    # Upload
    echo ""
    echo "Uploading compressed media files..."
    huggingface-cli upload "$REPO_ID" "$TEMP_VICAS" --repo-type dataset

    rm -rf "$TEMP_VICAS"
    echo "✓ Compressed media files uploaded"

    # Create extraction instructions
    cat > "/tmp/EXTRACTION_GUIDE.md" << 'EOF'
# Media Files Extraction Guide

The video frames and masks are provided as compressed archives to reduce download time.

## Extract all files

```bash
# Extract vicas data
cd gen_qa/vicas
tar -xzf video_frames.tar.gz
tar -xzf masks.tar.gz

# Extract st-evidence data
cd ../../gen_mask
tar -xzf video_frames_6fps.tar.gz
tar -xzf masks.tar.gz
```

## Quick start

```python
from huggingface_hub import hf_hub_download
import tarfile

# Download and extract
archive = hf_hub_download(
    repo_id="shijiewang/st-evidence-instruct",
    filename="gen_qa/vicas/video_frames.tar.gz",
    repo_type="dataset"
)

with tarfile.open(archive) as tar:
    tar.extractall(path="./data")
```
EOF

    huggingface-cli upload "$REPO_ID" /tmp/EXTRACTION_GUIDE.md --repo-type dataset
    rm /tmp/EXTRACTION_GUIDE.md
    ;;

  [Bb]*)
    echo ""
    echo "Uploading individual media files (this will take a while)..."

    TEMP_MEDIA="/tmp/st-evidence-media-$$"
    mkdir -p "$TEMP_MEDIA"

    # Copy with dereferenced symlinks
    rsync -avL --progress "$SOURCE_DIR/" "$TEMP_MEDIA/"

    echo ""
    echo "Total size:"
    du -sh "$TEMP_MEDIA"

    huggingface-cli upload "$REPO_ID" "$TEMP_MEDIA" --repo-type dataset

    rm -rf "$TEMP_MEDIA"
    echo "✓ All files uploaded"
    ;;

  *)
    echo "Skipping media files upload"
    ;;
esac

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "View your dataset at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"

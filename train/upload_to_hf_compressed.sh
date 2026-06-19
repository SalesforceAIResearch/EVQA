#!/bin/bash
# Upload st-evidence-instruct dataset as compressed archives
# Fast upload, users need to extract

set -e

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="shijiewang/st-evidence-instruct"
TEMP_DIR="/tmp/st-evidence-compressed-$$"

echo "============================================================"
echo "ST-Evidence-Instruct Compressed Upload to Hugging Face"
echo "============================================================"

mkdir -p "$TEMP_DIR"

# Copy metadata files directly
echo ""
echo "Step 1: Copying metadata files..."
rsync -av --progress \
  --include='*.csv' \
  --include='*.md' \
  --include='*.json' \
  --include='*.pkl' \
  --include='*/' \
  --exclude='*' \
  "$SOURCE_DIR/" "$TEMP_DIR/"

# Create compressed archives for media files
echo ""
echo "Step 2: Creating compressed archives (this may take a while)..."

mkdir -p "$TEMP_DIR/gen_qa/vicas"
mkdir -p "$TEMP_DIR/gen_mask"

# Compress vicas data
echo "  [1/4] Compressing vicas/video_frames..."
tar -czf "$TEMP_DIR/gen_qa/vicas/video_frames.tar.gz" \
  -C /fsx/home/shijie.wang/code/UniPixel/data/vicas \
  video_frames \
  --checkpoint=10000 \
  --checkpoint-action=echo='%{%Y-%m-%d %H:%M:%S}T: %u files archived'

echo "  [2/4] Compressing vicas/masks..."
tar -czf "$TEMP_DIR/gen_qa/vicas/masks.tar.gz" \
  -C /fsx/home/shijie.wang/code/UniPixel/data/vicas \
  masks

echo "  [3/4] Compressing st-evidence/video_frames_6fps..."
tar -czf "$TEMP_DIR/gen_mask/video_frames_6fps.tar.gz" \
  -C /fsx/home/shijie.wang/code/st-evidence/training_annotations \
  video_frames_6fps \
  --checkpoint=10000 \
  --checkpoint-action=echo='%{%Y-%m-%d %H:%M:%S}T: %u files archived'

echo "  [4/4] Compressing st-evidence/masks..."
tar -czf "$TEMP_DIR/gen_mask/masks.tar.gz" \
  -C /fsx/home/shijie.wang/code/st-evidence/training_annotations/mask_results/shared \
  masks_all_clevrer_gdino_refined

echo "✓ Compression complete"

# Create extraction guide
echo ""
echo "Step 3: Creating extraction guide..."
cat > "$TEMP_DIR/EXTRACTION_GUIDE.md" << 'EOF'
# ST-Evidence-Instruct Dataset - Extraction Guide

This dataset contains compressed archives for faster download. You need to extract them before use.

## Dataset Structure

```
st-evidence-instruct/
├── gen_qa/vicas/
│   ├── st_evidence_vicas.csv          # Metadata (ready to use)
│   ├── video_frames.tar.gz            # Extract this
│   └── masks.tar.gz                   # Extract this
└── gen_mask/
    ├── st_evidence.csv                # Metadata (ready to use)
    ├── st_evidence_meta.pkl           # Metadata (ready to use)
    ├── video_frames_6fps.tar.gz       # Extract this
    └── masks.tar.gz                   # Extract this
```

## Quick Extraction

### Option 1: Extract all (Bash)

```bash
# After downloading the dataset
cd st-evidence-instruct

# Extract vicas data
cd gen_qa/vicas
tar -xzf video_frames.tar.gz
tar -xzf masks.tar.gz
cd ../..

# Extract st-evidence data
cd gen_mask
tar -xzf video_frames_6fps.tar.gz
tar -xzf masks.tar.gz
cd ..
```

### Option 2: Extract with Python

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import tarfile
import os

# Download dataset metadata
dataset = load_dataset("shijiewang/st-evidence-instruct", data_files="gen_qa/vicas/st_evidence_vicas.csv")

# Download and extract archives
def extract_archive(repo_id, filename, extract_to):
    archive_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )

    print(f"Extracting {filename}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"✓ Extracted to {extract_to}")

# Extract all archives
repo_id = "shijiewang/st-evidence-instruct"
base_dir = "./st-evidence-instruct"

os.makedirs(f"{base_dir}/gen_qa/vicas", exist_ok=True)
os.makedirs(f"{base_dir}/gen_mask", exist_ok=True)

extract_archive(repo_id, "gen_qa/vicas/video_frames.tar.gz", f"{base_dir}/gen_qa/vicas")
extract_archive(repo_id, "gen_qa/vicas/masks.tar.gz", f"{base_dir}/gen_qa/vicas")
extract_archive(repo_id, "gen_mask/video_frames_6fps.tar.gz", f"{base_dir}/gen_mask")
extract_archive(repo_id, "gen_mask/masks.tar.gz", f"{base_dir}/gen_mask")
```

## Selective Extraction

If you only need specific data:

```bash
# Only vicas data
tar -xzf gen_qa/vicas/video_frames.tar.gz
tar -xzf gen_qa/vicas/masks.tar.gz

# Only st-evidence data
tar -xzf gen_mask/video_frames_6fps.tar.gz
tar -xzf gen_mask/masks.tar.gz
```

## After Extraction

The directory structure will match the original:

```
gen_qa/vicas/
├── video_frames/
│   ├── video1/
│   ├── video2/
│   └── ...
└── masks/
    ├── video1/
    └── ...
```

Then you can use the dataset as documented in `README.md` and `USAGE_EXAMPLE.md`.
EOF

echo "✓ Extraction guide created"

# Show final sizes
echo ""
echo "Step 4: Final archive sizes:"
echo "------------------------------------------------------------"
du -sh "$TEMP_DIR"/*/*/*tar.gz
echo ""
echo "Total upload size:"
du -sh "$TEMP_DIR"

# Upload to Hugging Face
echo ""
echo "Step 5: Uploading to Hugging Face..."
echo "------------------------------------------------------------"

huggingface-cli upload "$REPO_ID" "$TEMP_DIR" --repo-type dataset

# Cleanup
echo ""
echo "Step 6: Cleaning up..."
rm -rf "$TEMP_DIR"

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "View your dataset at: https://huggingface.co/datasets/$REPO_ID"
echo ""
echo "Note: Users will need to extract archives before use."
echo "See EXTRACTION_GUIDE.md in the dataset for instructions."
echo "============================================================"

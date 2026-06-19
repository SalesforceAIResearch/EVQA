#!/bin/bash
# Upload selected files from st-evidence-instruct dataset
# Only uploads compressed archives, CSV, and PKL files

set -e

# Activate conda environment
source ~/micromamba/etc/profile.d/conda.sh
conda activate h200_clean

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="Salesforce/ST-Evidence-Instruct"
TEMP_DIR="/tmp/st-evidence-upload-$$"

echo "============================================================"
echo "ST-Evidence-Instruct Selected Files Upload"
echo "============================================================"

# Create temp directory structure
echo ""
echo "Step 1: Preparing files..."
mkdir -p "$TEMP_DIR/gen_mask"
mkdir -p "$TEMP_DIR/gen_qa/vicas"

# Copy gen_mask files
echo "  Copying gen_mask files..."
cp "$SOURCE_DIR/gen_mask/masks.tar.gz" "$TEMP_DIR/gen_mask/"
cp "$SOURCE_DIR/gen_mask/video_frames_6fps.tar.gz" "$TEMP_DIR/gen_mask/"
cp -L "$SOURCE_DIR/gen_mask/st_evidence.csv" "$TEMP_DIR/gen_mask/"
cp -L "$SOURCE_DIR/gen_mask/st_evidence_meta.pkl" "$TEMP_DIR/gen_mask/"

# Copy gen_qa files
echo "  Copying gen_qa files..."
cp "$SOURCE_DIR/gen_qa/vicas/st_evidence_vicas.csv" "$TEMP_DIR/gen_qa/vicas/"

# Copy README and documentation
echo "  Copying documentation..."
cp "$SOURCE_DIR/README.md" "$TEMP_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/USAGE_EXAMPLE.md" "$TEMP_DIR/" 2>/dev/null || true

# Create extraction guide
echo "  Creating extraction guide..."
cat > "$TEMP_DIR/EXTRACTION_GUIDE.md" << 'EOF'
# ST-Evidence-Instruct Dataset - Quick Start

## Dataset Structure

```
st-evidence-instruct/
├── gen_mask/
│   ├── masks.tar.gz              # Compressed masks (extract this)
│   ├── video_frames_6fps.tar.gz  # Compressed video frames (extract this)
│   ├── st_evidence.csv           # Metadata (ready to use)
│   └── st_evidence_meta.pkl      # Metadata (ready to use)
└── gen_qa/vicas/
    └── st_evidence_vicas.csv     # QA pairs with temporal evidence (ready to use)
```

## Quick Start

### 1. Download the dataset

```python
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="Salesforce/ST-Evidence-Instruct",
    repo_type="dataset"
)
print(f"Dataset downloaded to: {dataset_path}")
```

### 2. Extract compressed files

```bash
cd st-evidence-instruct/gen_mask

# Extract masks
tar -xzf masks.tar.gz

# Extract video frames
tar -xzf video_frames_6fps.tar.gz
```

Or use Python:

```python
import tarfile
from pathlib import Path

def extract_archive(archive_path, extract_to):
    print(f"Extracting {archive_path}...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"✓ Extracted to {extract_to}")

# Extract gen_mask archives
gen_mask_dir = Path(dataset_path) / "gen_mask"
extract_archive(gen_mask_dir / "masks.tar.gz", gen_mask_dir)
extract_archive(gen_mask_dir / "video_frames_6fps.tar.gz", gen_mask_dir)
```

### 3. Load the data

```python
import pandas as pd
import pickle

# Load gen_mask data
gen_mask_csv = pd.read_csv(f"{dataset_path}/gen_mask/st_evidence.csv")
with open(f"{dataset_path}/gen_mask/st_evidence_meta.pkl", 'rb') as f:
    gen_mask_meta = pickle.load(f)

# Load gen_qa data
gen_qa_csv = pd.read_csv(f"{dataset_path}/gen_qa/vicas/st_evidence_vicas.csv")

print(f"gen_mask samples: {len(gen_mask_csv)}")
print(f"gen_qa samples: {len(gen_qa_csv)}")
```

## After Extraction

The directory structure will be:

```
gen_mask/
├── masks/               # Extracted
│   ├── video1/
│   ├── video2/
│   └── ...
├── video_frames_6fps/   # Extracted
│   ├── video1/
│   ├── video2/
│   └── ...
├── st_evidence.csv
└── st_evidence_meta.pkl
```

## Data Format

### gen_mask/st_evidence.csv

Contains video QA samples with spatial-temporal evidence:
- `video_id`: Video identifier
- `question`: Question text
- `answer`: Answer text
- `mask_evidence`: Spatial mask references
- `temporal_evidence`: Temporal segments

### gen_qa/vicas/st_evidence_vicas.csv

Contains 141k QA pairs generated from ViCaS videos:
- `entry_id`: Unique identifier
- `video_id`: Video identifier
- `question`: Question text
- `answer`: Answer text
- `candidates`: Multiple choice options
- `mask_evidence`: Spatial evidence
- `temporal_evidence`: Temporal segments

For more details, see README.md and USAGE_EXAMPLE.md.
EOF

echo "✓ Files prepared"

# Show what we're uploading
echo ""
echo "Step 2: Upload summary"
echo "------------------------------------------------------------"
du -sh "$TEMP_DIR"/*
echo ""
echo "Total size:"
du -sh "$TEMP_DIR"
echo "------------------------------------------------------------"

# Upload to Hugging Face
echo ""
echo "Step 3: Uploading to Hugging Face..."
echo "Repository: $REPO_ID"
echo ""

huggingface-cli upload "$REPO_ID" "$TEMP_DIR" --repo-type dataset

# Cleanup
echo ""
echo "Step 4: Cleaning up..."
rm -rf "$TEMP_DIR"

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo ""
echo "Uploaded files:"
echo "  - gen_mask/masks.tar.gz (253MB)"
echo "  - gen_mask/video_frames_6fps.tar.gz (2.2GB)"
echo "  - gen_mask/st_evidence.csv"
echo "  - gen_mask/st_evidence_meta.pkl"
echo "  - gen_qa/vicas/st_evidence_vicas.csv"
echo "  - Documentation files"
echo ""
echo "View at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"

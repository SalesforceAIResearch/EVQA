#!/bin/bash
# Direct upload to Hugging Face without copying to temp

set -e

# Activate conda environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate h200_clean

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="Salesforce/ST-Evidence-Instruct"

echo "============================================================"
echo "ST-Evidence-Instruct Direct Upload"
echo "============================================================"
echo "Repository: $REPO_ID"
echo ""

# Upload gen_mask files
echo "Step 1: Uploading gen_mask files..."
huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/masks.tar.gz" \
  gen_mask/masks.tar.gz \
  --repo-type dataset

huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/video_frames_6fps.tar.gz" \
  gen_mask/video_frames_6fps.tar.gz \
  --repo-type dataset

huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence.csv" \
  gen_mask/st_evidence.csv \
  --repo-type dataset

huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence_meta.pkl" \
  gen_mask/st_evidence_meta.pkl \
  --repo-type dataset

echo "✓ gen_mask files uploaded"

# Upload gen_qa files
echo ""
echo "Step 2: Uploading gen_qa files..."
huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/gen_qa/vicas/st_evidence_vicas.csv" \
  gen_qa/vicas/st_evidence_vicas.csv \
  --repo-type dataset

echo "✓ gen_qa files uploaded"

# Upload documentation
echo ""
echo "Step 3: Uploading documentation..."
huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/README.md" \
  README.md \
  --repo-type dataset

huggingface-cli upload "$REPO_ID" \
  "$SOURCE_DIR/USAGE_EXAMPLE.md" \
  USAGE_EXAMPLE.md \
  --repo-type dataset

echo "✓ Documentation uploaded"

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "View at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"

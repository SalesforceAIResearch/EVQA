#!/bin/bash
# Check token and upload

set -e

# Activate conda environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate h200_clean

echo "============================================================"
echo "Checking Hugging Face Authentication"
echo "============================================================"

# Check current auth
echo "Current user:"
hf auth whoami

echo ""
echo "============================================================"
echo "If you see a token permission error, your token might not"
echo "have WRITE permissions. You need to:"
echo "1. Go to https://huggingface.co/settings/tokens"
echo "2. Create a new token with WRITE permission"
echo "3. Run: hf auth login"
echo "============================================================"
echo ""
read -p "Press Enter to continue with upload, or Ctrl+C to abort and re-login..."

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="Salesforce/ST-Evidence-Instruct"

echo ""
echo "============================================================"
echo "Starting Upload to $REPO_ID"
echo "============================================================"

# Upload gen_mask files
echo ""
echo "[1/7] Uploading gen_mask/masks.tar.gz (253MB)..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/masks.tar.gz" \
  gen_mask/masks.tar.gz \
  --repo-type dataset

echo "[2/7] Uploading gen_mask/video_frames_6fps.tar.gz (44GB)..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/video_frames_6fps.tar.gz" \
  gen_mask/video_frames_6fps.tar.gz \
  --repo-type dataset

echo "[3/7] Uploading gen_mask/st_evidence.csv..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence.csv" \
  gen_mask/st_evidence.csv \
  --repo-type dataset

echo "[4/7] Uploading gen_mask/st_evidence_meta.pkl..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence_meta.pkl" \
  gen_mask/st_evidence_meta.pkl \
  --repo-type dataset

# Upload gen_qa files
echo "[5/7] Uploading gen_qa/vicas/st_evidence_vicas.csv..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_qa/vicas/st_evidence_vicas.csv" \
  gen_qa/vicas/st_evidence_vicas.csv \
  --repo-type dataset

# Upload documentation
echo "[6/7] Uploading README.md..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/README.md" \
  README.md \
  --repo-type dataset

echo "[7/7] Uploading USAGE_EXAMPLE.md..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/USAGE_EXAMPLE.md" \
  USAGE_EXAMPLE.md \
  --repo-type dataset

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "View at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"

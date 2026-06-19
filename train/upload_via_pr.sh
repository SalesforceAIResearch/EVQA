#!/bin/bash
# Upload to Salesforce repo via Pull Request

set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate h200_clean

SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
REPO_ID="Salesforce/ST-Evidence-Instruct"

echo "============================================================"
echo "Uploading to $REPO_ID via Pull Request"
echo "============================================================"
echo ""
echo "Note: Since you don't have direct write access, files will"
echo "be uploaded to a Pull Request for admin review."
echo ""

# Upload gen_mask files via PR
echo "[1/7] Uploading gen_mask/masks.tar.gz (253MB)..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/masks.tar.gz" \
  gen_mask/masks.tar.gz \
  --repo-type dataset \
  --create-pr

echo ""
echo "[2/7] Uploading gen_mask/video_frames_6fps.tar.gz (44GB)..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/video_frames_6fps.tar.gz" \
  gen_mask/video_frames_6fps.tar.gz \
  --repo-type dataset \
  --create-pr

echo ""
echo "[3/7] Uploading gen_mask/st_evidence.csv..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence.csv" \
  gen_mask/st_evidence.csv \
  --repo-type dataset \
  --create-pr

echo ""
echo "[4/7] Uploading gen_mask/st_evidence_meta.pkl..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_mask/st_evidence_meta.pkl" \
  gen_mask/st_evidence_meta.pkl \
  --repo-type dataset \
  --create-pr

echo ""
echo "[5/7] Uploading gen_qa/vicas/st_evidence_vicas.csv..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/gen_qa/vicas/st_evidence_vicas.csv" \
  gen_qa/vicas/st_evidence_vicas.csv \
  --repo-type dataset \
  --create-pr

echo ""
echo "[6/7] Uploading README.md..."
hf upload "$REPO_ID" \
  "$SOURCE_DIR/README.md" \
  README.md \
  --repo-type dataset \
  --create-pr

echo ""
echo "============================================================"
echo "✓ Upload via Pull Request initiated!"
echo ""
echo "A Pull Request has been created at:"
echo "https://huggingface.co/datasets/$REPO_ID/discussions"
echo ""
echo "Next steps:"
echo "1. Check the PR link above"
echo "2. Notify Salesforce admins to review and merge the PR"
echo "3. Once merged, the dataset will be in the main branch"
echo "============================================================"

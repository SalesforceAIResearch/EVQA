#!/bin/bash
# Upload st-evidence-instruct dataset to Hugging Face
# This script dereferences all symbolic links

set -e  # Exit on error

# Configuration
SOURCE_DIR="/fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct"
TEMP_DIR="/tmp/st-evidence-instruct-upload-$$"
REPO_ID="shijiewang/st-evidence-instruct"  # Change to your HF username

echo "============================================================"
echo "ST-Evidence-Instruct Dataset Upload to Hugging Face"
echo "============================================================"
echo "Source: $SOURCE_DIR"
echo "Temp dir: $TEMP_DIR"
echo "Target repo: $REPO_ID"
echo ""

# Step 1: Create temporary directory
echo "Step 1: Creating temporary directory..."
mkdir -p "$TEMP_DIR"
echo "✓ Created: $TEMP_DIR"

# Step 2: Copy data with rsync (dereference symlinks)
echo ""
echo "Step 2: Copying data (dereferencing symlinks with rsync)..."
echo "This may take a while for large datasets..."
rsync -avL --progress "$SOURCE_DIR/" "$TEMP_DIR/"
echo "✓ Data copy complete"

# Step 3: Show what we're uploading
echo ""
echo "Step 3: Checking directory structure..."
du -sh "$TEMP_DIR"/*

# Step 4: Upload to Hugging Face
echo ""
echo "Step 4: Uploading to Hugging Face..."
echo "Make sure you've run 'huggingface-cli login' first!"
echo ""

huggingface-cli upload "$REPO_ID" "$TEMP_DIR" --repo-type dataset

echo ""
echo "============================================================"
echo "✓ Upload complete!"
echo "View your dataset at: https://huggingface.co/datasets/$REPO_ID"
echo "============================================================"

# Step 5: Cleanup
echo ""
echo "Step 5: Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "✓ Cleanup complete"

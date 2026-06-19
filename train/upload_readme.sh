#!/bin/bash
set -e

eval "$(micromamba shell hook --shell bash)"
micromamba activate h200_clean

echo "Uploading README.md..."
hf upload wang-sj16/ST-Evidence-Instruct \
  /fsx/home/shijie.wang/code/EVQA/data/st-evidence-instruct/README.md \
  README.md \
  --repo-type dataset

echo "✓ README.md uploaded successfully!"
echo "View at: https://huggingface.co/datasets/wang-sj16/ST-Evidence-Instruct"

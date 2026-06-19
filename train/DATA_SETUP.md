# Data Setup Guide

This document explains how to set up the data directories for training.

## Quick Setup

Run the setup script to create all necessary symbolic links:

```bash
bash scripts/setup_data_links.sh
```

This will create symbolic links in the `data/` directory pointing to the actual datasets.

## Manual Setup

If you need to set up links manually or on a different machine:

### 1. Required Datasets

The training script uses the following datasets:

- `revos` - ReVOS video object segmentation
- `mevis` - MeViS video segmentation
- `lvvis` - LVVIS long video instance segmentation
- `ref_youtube_vos` - Referring YouTube-VOS
- `ref_davis17` - Referring DAVIS 2017
- `ref_sav` - Ref-SAV dataset
- `groundmore` - GroundMoRe grounding dataset
- `vicas` - ViCaS video understanding dataset
- `llava_instruct` - LLaVA instruction tuning data
- `videogpt_plus` - VideoGPT+ data
- `ViCaS` - ViCaS QA generation data
- `st-evidence` - Spatial-Temporal Evidence dataset

### 2. Create Symbolic Links

```bash
# Create data directory
mkdir -p data

# Link training datasets from UniPixel
ln -s ~/code/UniPixel/data/revos data/revos
ln -s ~/code/UniPixel/data/mevis data/mevis
ln -s ~/code/UniPixel/data/lvvis data/lvvis
ln -s ~/code/UniPixel/data/ref_youtube_vos data/ref_youtube_vos
ln -s ~/code/UniPixel/data/ref_davis17 data/ref_davis17
ln -s ~/code/UniPixel/data/ref_sav data/ref_sav
ln -s ~/code/UniPixel/data/groundmore data/groundmore
ln -s ~/code/UniPixel/data/vicas data/vicas
ln -s ~/code/UniPixel/data/llava_instruct data/llava_instruct
ln -s ~/code/UniPixel/data/videogpt_plus data/videogpt_plus

# Link special datasets
ln -s ~/code/ViCaS data/ViCaS
ln -s ~/code/st-evidence data/st-evidence
```

## Current Data Structure

After setup, your `data/` directory should look like:

```
data/
├── ViCaS -> /fsx/home/shijie.wang/code/ViCaS
├── st-evidence -> /fsx/home/shijie.wang/code/st-evidence
├── revos -> /fsx/home/shijie.wang/code/UniPixel/data/revos
├── mevis -> /fsx/home/shijie.wang/code/UniPixel/data/mevis
├── lvvis -> /fsx/home/shijie.wang/code/UniPixel/data/lvvis
├── ref_youtube_vos -> /fsx/home/shijie.wang/code/UniPixel/data/ref_youtube_vos
├── ref_davis17 -> /fsx/home/shijie.wang/code/UniPixel/data/ref_davis17
├── ref_sav -> /fsx/home/shijie.wang/code/UniPixel/data/ref_sav
├── groundmore -> /fsx/home/shijie.wang/code/UniPixel/data/groundmore
├── vicas -> /fsx/home/shijie.wang/code/UniPixel/data/vicas
├── llava_instruct -> /fsx/home/shijie.wang/code/UniPixel/data/llava_instruct
└── videogpt_plus -> /fsx/home/shijie.wang/code/UniPixel/data/videogpt_plus
```

## Verification

Check if all datasets are accessible:

```bash
ls -lh data/
```

All entries should show as symbolic links (`lrwxrwxrwx`) pointing to valid directories.

## Notes

- **Relative Paths**: All dataset loading code uses relative paths starting with `data/`
- **Portability**: On a different machine, simply adjust the symbolic link targets
- **Storage**: No data is duplicated; links point to the original locations
- **Training**: The training script `scripts/finetune_eccv_merged_v2.sh` expects this structure

## Troubleshooting

If you get "file not found" errors during training:

1. Check if symlinks exist: `ls -la data/`
2. Verify link targets exist: `ls -la ~/code/UniPixel/data/`
3. Re-run setup script: `bash scripts/setup_data_links.sh`

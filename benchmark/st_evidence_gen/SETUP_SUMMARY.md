# ST-Evidence Generation Benchmark - Setup Summary

## 目录结构
```
st_evidence_gen/
├── data/
│   ├── st_evidence_gen.csv          # Ground truth annotations
│   ├── videos_6fps/                 # Video files at 6fps
│   └── mask_annos_latest_img/       # GT masks (symlink)
├── results/                         # Auto-created during inference
│   ├── openai/                      # GPT model results
│   ├── gemini/                      # Gemini model results
│   ├── ours/                        # Our model results
│   ├── qwen3vl/                     # Qwen3VL results
│   └── ...
└── *_st_evidence.py                 # Model inference scripts
```

## 配置状态

### ✅ 已完成
- [x] 所有脚本使用统一的相对路径
- [x] 数据文件：`data/st_evidence_gen.csv`
- [x] 视频目录：`data/videos_6fps`
- [x] GT masks：`data/mask_annos_latest_img`
- [x] 删除冗余文件（st_evidence_final.csv）
- [x] SA2VA路径配置支持多种方式（环境变量/相对路径/绝对路径）

### 📋 可用脚本（10个模型）

1. **gemini_st_evidence.py** - Gemini models
2. **gpt_st_evidence.py** - GPT-4o/GPT-5
3. **internvl_3_5.py** - InternVL 3.5
4. **llava_ov1_5_st_evidence.py** - LLaVA-OV 1.5
5. **ours_st_evidence.py** - Our UniPixel model
6. **qwen2_5vl_st_evidence.py** - Qwen2.5-VL
7. **qwen3vl_st_evidence.py** - Qwen3-VL
8. **sa2va_st_evidence.py** - SA2VA
9. **unipixel_st_evidence.py** - UniPixel baseline
10. **videollama3_st_evidence.py** - VideoLLaMA3

### 📊 评估脚本

- **eval_st_evidence.py** - 核心评估脚本（QA + Temporal + Spatial）
- **eval_st_evidence_all.sh** - 批量评估脚本
- **eval_st_evidence_ours.sh** - Ours模型专用评估

### 🔧 辅助工具

- **ours_st_evidence_multigpu.sh** - 多GPU并行推理
- **unipixel_video_seg_multigpu.sh** - 视频分割多GPU

## 快速使用

### 1. 运行推理

```bash
cd /fsx/home/shijie.wang/code/EVQA/benchmark/st_evidence_gen

# 单GPU推理
python ours_st_evidence.py --model PolyU-ChenLab/UniPixel-3B --fps 1.0 --max-frames 128

# 多GPU推理（8个GPU）
bash ours_st_evidence_multigpu.sh PolyU-ChenLab/UniPixel-3B 8 0
```

### 2. 运行评估

```bash
# 基础评估
python eval_st_evidence.py --pred_file results/ours/model_1fps.json

# 包含mask评估
python eval_st_evidence.py \
    --pred_file results/ours/model_1fps.json \
    --eval_masks \
    --pred_mask_dir results/ours/model_1fps/
```

### 3. SA2VA特殊配置

SA2VA需要额外的依赖路径，有三种配置方式：

**方式1：环境变量（推荐）**
```bash
export SA2VA_PATH=/path/to/Sa2VA
python sa2va_st_evidence.py --model xxx
```

**方式2：相对路径**
将Sa2VA放在相对路径（自动查找）：
- `../../../Sa2VA`
- `../../Sa2VA`

**方式3：使用默认绝对路径**
脚本会fallback到原始路径：`/fsx/home/shijie.wang/code/Sa2VA`

## 输出格式

### 推理输出（JSON）
```json
{
  "entry_id": {
    "video_id": "xxx",
    "parsed_response": {
      "answer": "A",
      "segments": [[2.5, 5.8], [7.1, 9.4]],
      "mask_path": "results/ours/model/entry_id/masks/"
    }
  }
}
```

### 评估输出
```
Metric       Acc    mIoU   mIoP   t-mean   J      F      J&F
Score (%)    45.2   32.1   38.5   35.3     42.5   45.8   44.2
```

## 注意事项

1. 所有脚本都是self-contained，可以在当前目录独立运行
2. 结果会自动保存到对应的 `results/<model>/` 目录
3. 支持 `--resume` 参数从中断处继续
4. 多GPU脚本使用文件锁防止冲突
5. 确保 `data/videos_6fps/` 包含所有视频文件

## 常见问题

**Q: 找不到视频文件**
A: 检查 `data/videos_6fps/` 目录是否包含视频

**Q: SA2VA导入失败**
A: 设置环境变量 `export SA2VA_PATH=/path/to/Sa2VA`

**Q: 多GPU冲突**
A: 脚本已实现文件锁，不应出现冲突。如果有问题，检查是否有僵尸进程

**Q: 内存不足**
A: 减少 `--max-frames` 或 `--fps` 参数

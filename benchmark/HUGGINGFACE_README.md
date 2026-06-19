# ST-Evidence Benchmark Dataset

ST-Evidence is a comprehensive benchmark for evaluating Spatial-Temporal Evidence generation in video understanding. It contains two tasks: **Generation (Gen)** and **Multiple Choice Question (MCQ)**.

## Dataset Overview

- **Total Videos**: ~1,300 videos at 6fps
- **Annotations**: Question-Answer pairs with temporal segments and spatial masks
- **Tasks**: Generation and MCQ
- **Domains**: Diverse video content

## Files Structure

```
ST-Evidence/
├── st_evidence_gen/              # Generation Task
│   ├── st_evidence_gen.csv       # 924KB - Annotations (entry_id, question, answer, segments, etc.)
│   ├── videos_6fps.tar.gz        # 8.3GB - Video files at 6fps
│   └── masks.tar.gz              # 560MB - Ground truth spatial masks
│
└── st_evidence_mcq/              # Multiple Choice Question Task
    ├── st_evidence_mcq.csv       # 313KB - MCQ annotations
    ├── mask_options.json         # 575KB - Mask options metadata
    ├── temp_options.json         # 679KB - Temporal options metadata
    └── options.tar.gz            # 1.5GB - Pre-rendered option masks (1,298 entries)
```

## Generation Task (st_evidence_gen)

### Data Format

**st_evidence_gen.csv** contains the following columns:
- `entry_id`: Unique identifier for each question
- `video_id`: Video identifier
- `video_path`: Relative path to video file
- `question`: Question text
- `candidates`: List of answer options (for reference)
- `answer`: Ground truth answer
- `segment`: Temporal evidence segments [[start1, end1], [start2, end2], ...]

### Usage

```python
import pandas as pd
import tarfile

# Load annotations
df = pd.read_csv('st_evidence_gen.csv')

# Extract videos
with tarfile.open('videos_6fps.tar.gz', 'r:gz') as tar:
    tar.extractall('videos_6fps/')

# Extract ground truth masks
with tarfile.open('masks.tar.gz', 'r:gz') as tar:
    tar.extractall('masks/')
```

### Evaluation Metrics

- **QA Accuracy**: Percentage of correct answers
- **Temporal IoU**: Intersection over Union for temporal segments
  - mIoU, TIoU@0.3, TIoU@0.5
- **Temporal IoP**: Intersection over Prediction
  - mIoP, TIoP@0.3, TIoP@0.5
- **Spatial Quality** (if masks generated):
  - J score (Jaccard/IoU)
  - F score (contour-based)
  - J&F score (average)

## MCQ Task (st_evidence_mcq)

### Data Format

**st_evidence_mcq.csv** contains:
- `entry_id`: Unique identifier
- `video_id`: Video identifier
- `video_path`: Path to video
- `question`: Question text
- `candidates`: Answer options
- `answer`: Correct answer
- `segment`: Temporal evidence
- `mask_options`: Reference to mask options
- `temp_options`: Reference to temporal options

**mask_options.json**: Contains spatial mask options for each question
**temp_options.json**: Contains temporal segment options for each question
**options.tar.gz**: Pre-rendered mask visualizations for options (1,298 entries)

### Usage

```python
import json
import pandas as pd

# Load MCQ annotations
df = pd.read_csv('st_evidence_mcq.csv')

# Load options
with open('mask_options.json', 'r') as f:
    mask_options = json.load(f)

with open('temp_options.json', 'r') as f:
    temp_options = json.load(f)

# Extract option masks
with tarfile.open('options.tar.gz', 'r:gz') as tar:
    tar.extractall('options/')
```

### Evaluation Metrics

Same as Generation task, but with multiple-choice format.

## Download & Setup

### Using HuggingFace Hub

```python
from huggingface_hub import snapshot_download

# Download entire dataset
snapshot_download(
    repo_id="Salesforce/ST-Evidence",
    repo_type="dataset",
    local_dir="./st_evidence_data"
)
```

### Manual Download

1. Download all files from this repository
2. Extract compressed files:
   ```bash
   tar -xzf videos_6fps.tar.gz
   tar -xzf masks.tar.gz
   tar -xzf options.tar.gz
   ```

## Citation

If you use this dataset, please cite:

```bibtex
@article{st-evidence2025,
  title={ST-Evidence: A Benchmark for Spatial-Temporal Evidence in Video Understanding},
  author={Wang, Shijie and others},
  year={2025}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact: [your contact info]

## Version

- **Version**: 1.0
- **Release Date**: 2025-03-14
- **Total Size**: ~10.4 GB (compressed)

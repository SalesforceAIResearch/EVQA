# Training UniPixel

## 🛠️ Environment Setup

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 12.8
- Python 3.12.11
- PyTorch 2.7.1
- [Transformers](https://github.com/huggingface/transformers) 4.53.3
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) 0.17.4
- [NNCore](https://github.com/yeliudev/nncore) 0.4.7

### Install the environment

1. Clone the repository from GitHub.

```shell
git clone https://github.com/PolyU-ChenLab/UniPixel.git
cd UniPixel
```

2. Setup the virtual environment.

```shell
conda create -n unipixel python=3.12 -y
conda activate unipixel

# you may modify 'cu128' to your own CUDA version
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# other versions have no been verified
pip install flash_attn==2.8.2 --no-build-isolation
```

3. Install dependencies.

```shell
pip install -r requirements.txt
```

For NPU users, please install the CPU version of PyTorch and [`torch_npu`](https://github.com/Ascend/pytorch) instead.

### Prepare base models

Download [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), then place them into the `model_zoo` folder.

```
UniPixel
└─ model_zoo
   ├─ Qwen2.5-VL-3B-Instruct
   └─ Qwen2.5-VL-7B-Instruct
```

## 📦 Dataset Preparation

We release all the datasets and benchmarks in this project on [Hugging Face](https://huggingface.co/datasets/PolyU-ChenLab/UniPixel-SFT-1M). After downloading the required datasets, extract the `tar.gz` files (no need to modify the relevant paths) and place them in the `data` folder. The processed files should be organized in the following structure (taking `ref_youtube_vos` as an example).

```
UniPixel
└─ data
   └─ ref_youtube_vos
      ├─ meta_expressions
      ├─ train
      ├─ valid
      └─ mask_dict.pkl
```

## 🔮 Start Training

Use the following commands to train UniPixel. Our experiments were conducted on **8 NVIDIA RTX 6000 Ada (48G) GPUs**. You may modify `nproc_per_node`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to keep the same global batch size (256 for stage 1 and 2, 32 for stage 3) if you have different device configurations.

```shell
# Launch full training of UniPixel-3B
bash scripts/launch_3b.sh

# Launch full training of UniPixel-7B
bash scripts/launch_7b.sh
```

The training logs and checkpoints will be saved in the `work_dirs` folder. After training all the roles, you may run the following script for auto evaluation.

```shell
bash scripts/auto_eval.sh <path-to-checkpoint>
```

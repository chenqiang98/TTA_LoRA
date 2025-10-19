# TTA_LoRA Robustness Evaluation

## MLLM Evaluation Part
This repository contains a flexible script (`run_evaluation.py`) for evaluating the robustness of Vision-Language Models (VLMs) on the ImageNet-C dataset and its variants.

### Features

- **Multiple Dataset Formats**: Seamlessly load datasets from local folders, `.tar` webdataset files, or directly from the Hugging Face Hub.
- **Automated Logging**: Automatically save detailed experiment results, including parameters, performance metrics, and the exact prompts used, to a structured JSON file in the `./results/` directory.
- **Flexible Evaluation**: Easily configure the model, dataset, number of samples, and specific tasks (classification vs. corruption detection) via command-line arguments.
- **Prompt Engineering Ready**: Log files include the prompts used, making it easy to analyze their impact and experiment with new ones.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/LIUUUUUZ/TTA_LoRA.git
    cd TTA_LoRA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    Ensure the necessary metadata files (`imagenet_class_index.json` and `corruption_index.json`) are present in the `./data/` directory. If you are using a local dataset, place it in the appropriate path (e.g., `./data/nano-imagenet-c`).

### Usage

The main script is `run_evaluation.py`. All parameters can be configured via command-line arguments.

#### Examples

Here are some common use cases:

**1. Quick Sanity Check**
(Evaluates a few samples from the default nano dataset to ensure the pipeline works)
```bash
python run_evaluation.py --quick_validate
```

**2. Evaluate a Specific Model on a Local Dataset**
```bash
python run_evaluation.py \
    --dataset_path ./data/nano-imagenet-c \
    --model_name HuggingFaceTB/SmolVLM-500M-Instruct
```

**3. Evaluate the Default Model on a Dataset from Hugging Face Hub**
```bash
python run_evaluation.py --dataset_path niuniandaji/nano-imagenet-c
```

**4. Evaluate a Specific Task and Limit Sample Size**
```bash
python run_evaluation.py \
    --dataset_path ./data/nano-imagenet-c \
    --task classification \
    --num_samples 100
```

#### Output

For each run, a JSON log file will be automatically generated in the `./results` directory. The filename is created based on the model and dataset used (e.g., `InternVL3-1B_on_nano_log.json`).

Each log file contains a list of experiments, where each entry includes:
- `timestamp`: The time of the experiment.
- `command`: The exact command used to run the evaluation, for easy reproducibility.
- `parameters`: A complete dictionary of all arguments used for the run.
- `results`: The final accuracy scores for the evaluated tasks.
- `prompts`: The exact prompt templates used for classification and corruption detection.




## MCDL-TTA Training Part

This section covers training LoRA (Low-Rank Adaptation) modules for Test-Time Adaptation (TTA) on corrupted images. The script (`main.py`) automatically clusters samples by corruption type and trains specialized LoRA modules for each corruption.

### Features

- **Corruption-Specific LoRA Training**: Trains separate LoRA modules for each corruption type detected by the MLLM.
- **Entropy Minimization**: Uses entropy minimization as the training objective for unsupervised adaptation.
- **Multiple Model Support**: Supports ResNet50 and ViT-Base architectures.
- **Flexible Configuration**: Easily adjust LoRA rank, learning rate, number of epochs, and other hyperparameters.

### Usage

The training task is performed using `main.py` with `--task train`.

#### Examples

**1. Train LoRA Modules for ResNet50**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name resnet50 \
    --task train \
    --num_epochs 1 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --lora_rank 5
```

**2. Train LoRA Modules for ViT-Base**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name vit_base \
    --task train \
    --num_epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --lora_rank 8
```

**3. Train with Limited Samples**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name resnet50 \
    --task train \
    --num_samples 1000 \
    --num_epochs 1
```

#### Output

Trained LoRA modules are automatically saved to the `./LoRA/` directory. A timestamped subdirectory is created for each training run (e.g., `LoRA/resnet50_20251019_234132/`).

Each subdirectory contains:
- `{corruption_name}_lora.pth`: Individual LoRA modules for each corruption type (e.g., `brightness_lora.pth`, `fog_lora.pth`).

Each `.pth` file includes:
- `corruption_name`: The name of the corruption type.
- `lora_rank`: The rank used for the LoRA matrices.
- `lora_modules`: State dictionaries of the LoRA parameters for each target module.

---

## MCDL-TTA Evaluation Part

This section covers evaluating the trained LoRA modules on test samples. The script dynamically loads and merges LoRA modules based on the top-k corruption predictions for each sample.

### Features

- **Dynamic LoRA Merging**: Automatically loads and merges multiple LoRA modules based on top-k corruption predictions.
- **Efficient Caching**: Caches merged LoRA modules to avoid redundant computations.
- **Flexible Top-K Selection**: Configure how many corruption-specific LoRA modules to use per sample.
- **Comprehensive Metrics**: Reports overall accuracy and detailed statistics.

### Usage

The evaluation task is performed using `main.py` with `--task eval`.

#### Examples

**1. Evaluate with Top-1 Corruption LoRA (Single LoRA per Sample)**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name resnet50 \
    --task eval \
    --lora_load_dir LoRA/resnet50_20251019_234132 \
    --eval_top_k_corruptions 1
```

**2. Evaluate with Top-3 Corruption LoRAs (Merge 3 LoRAs per Sample)**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name resnet50 \
    --task eval \
    --lora_load_dir LoRA/resnet50_20251019_234132 \
    --eval_top_k_corruptions 3 \
    --num_samples 500
```

**3. Full Evaluation with Top-5 Corruption Predictions**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --top_k_values 5 \
    --num_samples 500 \
    --prediction_model_name resnet50 \
    --task eval \
    --lora_load_dir LoRA/resnet50_20251019_234132 \
    --eval_top_k_corruptions 3
```

**4. Evaluate ViT-Base Model**
```bash
python main.py \
    --dataset_path ./data/nano-imagenet-c \
    --prediction_model_name vit_base \
    --task eval \
    --lora_load_dir LoRA/vit_base_20251018_222151 \
    --eval_top_k_corruptions 2 \
    --batch_size 16
```

#### Key Parameters

- `--lora_load_dir`: Path to the directory containing trained LoRA modules (required for evaluation).
- `--eval_top_k_corruptions`: Number of top corruption predictions to use for each sample (default: 1).
- `--num_samples`: Number of samples to evaluate (useful for quick testing).
- `--top_k_values`: Number of top-k corruption predictions from MLLM to consider (affects preprocessing).

#### Output

The evaluation script outputs:
- **Total samples**: Number of samples evaluated.
- **Correct predictions**: Number of correct classifications.
- **Accuracy**: Overall classification accuracy (%).
- **Number of cached merged LoRA modules**: Indicates how many unique corruption combinations were encountered.
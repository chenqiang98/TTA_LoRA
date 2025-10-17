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

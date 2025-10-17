# Models in MCDL-TTA
        
## MLLM(args) function in MCDL-TTA

### Inputs (args)
- `model_name` (str): Hugging Face model id (e.g., `OpenGVLab/InternVL3-1B`).
- `dataset_path` (str): Path or id to the dataset (folder, `.tar`, or HF repo id).
- `num_samples` (int|None): Number of images to evaluate; uses full dataset if None.
- `batch_size` (int): DataLoader/inference batch size.
- `num_workers` (int): DataLoader worker count.
- `top_k_values` (int|None): If set, returns Top-K corruption predictions per image.
- `quick_validate` (flag): If true, runs a fast sanity check on a small subset.
- `device` (str|None): `cuda`, `cpu`, etc. Auto-detected if None.
- `seed` (int): Random seed for reproducibility.

### Outputs
Returns a PyTorch `Dataset` where each item is:
```
image, labels = dataset[idx]

# labels is a tuple:
# (
#   corruption_label: int,    # ground-truth corruption index
#   class_label: int,         # ground-truth class index
#   severity_label: int,      # corruption severity level
#   top_k_predictions: List[str]  # model-predicted Top-K corruption names
# )
```

### How It Works (high level)
1. Seeds RNGs and selects device/dtype automatically.
2. Loads the specified VLM and tokenizer from Hugging Face.
3. Loads dataset (folder/WebDataset/HF Hub) and ImageNet/Corruption metadata.
4. Collects images and labels; dynamically patches each image to 448×448 crops.
5. Prompts the VLM to identify corruption types (Top-1 or Top-K).
6. Builds and returns an augmented dataset with predicted Top-K corruption names.

### Minimal Example
```python
import argparse
from model.MLLM import MLLM

args = argparse.Namespace(
    model_name="OpenGVLab/InternVL3-1B",
    dataset_path="./data/nano-imagenet-c",
    num_samples=None,      # evaluate all samples
    batch_size=32,
    num_workers=4,
    top_k_values=5,        # return Top-5 corruption predictions per image
    quick_validate=False,
    device=None,           # auto-detect
    seed=7600,
)

augmented_dataset = MLLM(args)

# Example: inspect one sample
image, (corr_idx, cls_idx, sev_idx, topk_names) = augmented_dataset[0]
print(topk_names)  # e.g., ["gaussian_noise", "shot_noise", ...]
```

### Notes
- Images are dynamically split into patches before inference for better coverage.
- When `top_k_values` is None, the model is prompted for a single best corruption type.
- Use `--quick_validate` during development to iterate quickly.

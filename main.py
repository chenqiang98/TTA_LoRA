import argparse
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


from model.MLLM import MLLM
from model.resnet import resnet50, resnet18, get_resnet_train_transform, get_resnet_eval_transform
from model.vit import vit_base_patch16_224, get_vit_train_transform, get_vit_eval_transform


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""
    def __init__(self, in_features, out_features, rank=5):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        # x shape: (batch_size, ..., in_features)
        # We need to apply matrix multiplication on the last dimension
        return x @ self.lora_A @ self.lora_B


class CorruptionDataset(Dataset):
    """Dataset wrapper for a specific corruption type"""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image, label = self.samples[idx]
        corruption_id, cls, severity, topk_corruptions = label
        if self.transform:
            image = self.transform(image)
        return image, cls


def safe_pil_collate(batch):
    """A collate function that filters out None values and returns lists."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return [], []
    images, labels = zip(*batch)
    return list(images), list(labels)


def get_target_modules(model, model_name):
    """Get the target modules to apply LoRA based on model type"""
    target_modules = {}
    
    if model_name == 'resnet50':
        # For ResNet50: Apply LoRA to the last two FC layers
        # The last FC is model.fc, and we need to find the second-to-last FC
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Typically ResNet50 only has one FC layer at the end (model.fc)
                # If we want the last two FC layers, we need to check the architecture
                # For standard ResNet50, we'll apply to the final fc layer
                if 'fc' in name:
                    target_modules[name] = module
    
    elif model_name == 'vit_base':
        # For ViT: Apply LoRA to all Q, K, V projections and last two FC layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Q, K, V projections in attention blocks
                if 'qkv' in name or 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    target_modules[name] = module
                # Last two FC layers (typically in the head)
                elif 'head' in name or 'mlp.fc' in name:
                    target_modules[name] = module
    
    return target_modules


def train_lora_for_corruption(prediction_model, corruption_samples, transform, args, corruption_name, device):
    """Train a LoRA module for a specific corruption type"""
    # Create dataset and dataloader
    corruption_dataset = CorruptionDataset(corruption_samples, transform)
    train_loader = DataLoader(
        corruption_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Get target modules based on model type
    target_modules = get_target_modules(prediction_model, args.prediction_model_name)
    
    # Add LoRA layers to the target modules
    lora_modules = {}
    for name, module in target_modules.items():
        in_features = module.in_features
        out_features = module.out_features
        lora_modules[name] = LoRALayer(in_features, out_features, rank=args.lora_rank).to(device)
    
    print(f"Applied LoRA to {len(lora_modules)} modules: {list(lora_modules.keys())}")
    
    # Only optimize LoRA parameters
    lora_params = []
    for lora_module in lora_modules.values():
        lora_params.extend(list(lora_module.parameters()))
    
    optimizer = optim.Adam(lora_params, lr=args.learning_rate)
    
    # Training loop
    prediction_model.eval()  # Keep base model frozen
    for lora_module in lora_modules.values():
        lora_module.train()
    
    # Register forward hooks to apply LoRA
    hooks = []
    
    def create_lora_hook(module_name, lora_module):
        def hook(module, input, output):
            # Apply LoRA to the input instead of output
            # input is a tuple, so we get the first element
            input_tensor = input[0]
            lora_delta = lora_module(input_tensor)
            return output + lora_delta
        return hook
    
    # Register hooks for all target modules
    for name, module in target_modules.items():
        if name in lora_modules:
            hook = module.register_forward_hook(create_lora_hook(name, lora_modules[name]))
            hooks.append(hook)
    
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(train_loader, desc=f"Training LoRA for {corruption_name} - Epoch {epoch+1}/{args.num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with LoRA (hooks will apply LoRA automatically)
            lora_output = prediction_model(images)
            
            # Compute entropy loss (minimize prediction entropy)
            probs = torch.softmax(lora_output, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            loss = entropy.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Entropy Loss: {avg_loss:.4f}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return lora_modules


def main(args):
    # Set device
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # load corruption and class index
    metadata_source_path = "data"
    class_index_path = os.path.join(metadata_source_path, 'imagenet_class_index.json')
    corruption_index_path = os.path.join(metadata_source_path, 'corruption_index.json')
    
    # Load corruption index to get the mapping
    import json
    with open(corruption_index_path, 'r') as f:
        corruption_index = json.load(f)
    
    # get processed dataset(with Topk corruption as labels) 
    dataset = MLLM(args)

    test_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=safe_pil_collate
    )

    # Initialize dictionary to store datasets by corruption type
    corruption_datasets = {}
    for corruption_name in corruption_index.keys():
        corruption_datasets[corruption_name] = []
    # Iterate through all data and cluster by top1 corruption
    for batch_images, batch_labels in tqdm(test_loader, desc="Clustering by corruption"):
        for i, (image, label) in enumerate(zip(batch_images, batch_labels)):
            corruption_id, cls, severity, topk_corruptions = label
            # Get the top1 corruption name (first in the topk list)
            top1_corruption = topk_corruptions[0]
            # Add the sample to the corresponding corruption dataset
            corruption_datasets[top1_corruption].append((image, label))

    # Print statistics
    print("Dataset clustering by top1 corruption:")
    for corruption_name, samples in corruption_datasets.items():
        print(f"{corruption_name}: {len(samples)} samples")

    # load prediction model and transform
    if args.prediction_model_name == 'resnet50':
        prediction_model = resnet50(pretrained=True)
        prediction_transform = get_resnet_train_transform()
    elif args.prediction_model_name == 'vit_base':
        prediction_model = vit_base_patch16_224(pretrained=True)
        prediction_transform = get_vit_train_transform()
    else:
        raise ValueError(f"Invalid prediction model name: {args.prediction_model_name}")
    
    prediction_model = prediction_model.to(device)
    
    if args.task == 'train':
      # Create timestamp for saving LoRA modules
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      lora_save_dir = os.path.join(args.lora_save_dir, f"{args.prediction_model_name}_{timestamp}")
      os.makedirs(args.lora_save_dir, exist_ok=True)
      os.makedirs(lora_save_dir, exist_ok=True)
      print(f"LoRA modules will be saved to: {lora_save_dir}")
      
      # Train LoRA for each corruption type
      for corruption_name, samples in corruption_datasets.items():
         if len(samples) == 0:
               print(f"Skipping {corruption_name} (no samples)")
               continue
         
         print(f"\n{'='*80}")
         print(f"Training LoRA for corruption: {corruption_name}")
         print(f"{'='*80}")
         
         lora_modules = train_lora_for_corruption(
               prediction_model, 
               samples, 
               prediction_transform, 
               args, 
               corruption_name,
               device
         )
         
         # Save LoRA modules for this corruption
         corruption_save_path = os.path.join(lora_save_dir, f"{corruption_name}_lora.pth")
         torch.save({
               'corruption_name': corruption_name,
               'lora_rank': args.lora_rank,
               'lora_modules': {name: module.state_dict() for name, module in lora_modules.items()}
         }, corruption_save_path)
         print(f"Saved LoRA modules to: {corruption_save_path}")
      
      print(f"\n{'='*80}")
      print(f"All LoRA modules saved to: {lora_save_dir}")
      print(f"{'='*80}")

      

if __name__ == "__main__":
    """
    Example Usage:
    ------------------------------------------------------------------------------------
    
    1. Quick Sanity Check (evaluates a few samples from the nano dataset):
       python run_evaluation.py --quick_validate
    
    2. Evaluate a specific model on the full local 'nano-imagenet-c' dataset:
       python run_evaluation.py \
           --dataset_path ./data/nano-imagenet-c \
           --model_name HuggingFaceTB/SmolVLM-500M-Instruct

    3. Evaluate the default model on the 'nano-imagenet-c' dataset from Hugging Face Hub:
       python run_evaluation.py --dataset_path niuniandaji/nano-imagenet-c
    
    4. Evaluate only the 'classification' task and limit to 100 samples:
       python run_evaluation.py \
           --dataset_path ./data/nano-imagenet-c \
           --task classification \
           --num_samples 100
    
    ------------------------------------------------------------------------------------
    
    Notes:
    - Results are automatically saved to a JSON file in the './results' directory.
    - The log file is named based on the model and dataset used.
    - Metadata files (e.g., class indices) are expected in the './data' directory.

    Available Models & Datasets:
    - Models: This script is compatible with various Vision-Language Models from Hugging Face.
      Some tested models include:
      - OpenGVLab/InternVL3-1B
      - OpenGVLab/InternVL3_5-1B
      - OpenGVLab/InternVL3_5-2B
      - HuggingFaceTB/SmolVLM-256M-Instruct
      - HuggingFaceTB/SmolVLM-500M-Instruct

    - Datasets: The --dataset_path argument supports three formats:
      1. Path to a local folder (e.g., ./data/mini-ImageNet-C)
      2. Path to a local .tar webdataset file (e.g., ./data/nano-ImageNet-C.tar)
      3. A Hugging Face Hub dataset ID (e.g., niuniandaji/nano-imagenet-c)
    """
    
    parser = argparse.ArgumentParser(description="Evaluate Vision-Language Models on ImageNet-C")
    parser.add_argument("--dataset_path", type=str, default="./data/nano-imagenet-c", help="Path to the dataset (folder, .tar file, or HF repo ID).")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-2B", help="Model used in MLLM part")
    parser.add_argument("--prediction_model_name", type=str, default='resnet50',choices=['resnet50', 'vit_base'], help="Model used in classification prediction task.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of images to evaluate. If not provided, evaluates the entire dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--task", type=str, default="train", choices=['train', 'eval'], help="Task to perform.")
    parser.add_argument("--seed", type=int, default=7600, help="Random seed for reproducibility.")
    parser.add_argument("--quick_validate", action='store_true', help="Run in quick validation mode with a small subset of data.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computation (e.g., 'cpu', 'cuda'). Defaults to auto-detection.")
    parser.add_argument("--top_k_values", type=int, default=None, help="Top-k values to evaluate. If not provided, evaluates only top-1 accuracy.")

    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train the LoRA model.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the LoRA model.")
    parser.add_argument("--lora_rank", type=int, default=5, help="Rank of the LoRA model.")
    parser.add_argument("--lora_save_dir", type=str, default="LoRA", help="Directory to save the LoRA modules.")
    parser.add_argument("--lora_load_dir", type=str, default=None, help="Directory to load the LoRA modules.")
    
    args = parser.parse_args()
    main(args)

import argparse
import json
import math
import random
import numpy as np
import torch
import requests
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import sys
import datetime
import time

from data.dataloader import TTALoRADataset, worker_init_fn, WebTTALoRADataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def set_seed(seed=7600):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_pil_collate(batch):
    """A collate function that filters out None values and returns lists."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return [], []  # Return empty lists if the whole batch is invalid
    images, labels = zip(*batch)
    return list(images), list(labels)

def worker_init_fn(worker_id):
    """Worker init function for DataLoader to ensure reproducibility."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the closest aspect ratio from a list of target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamically preprocesses an image by splitting it into patches."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """Loads and preprocesses an image file."""
    if isinstance(image_file, str):
        if image_file.startswith('http'):
            image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def create_prompt(class_names, prompt_template, k_values=None):
    """Creates a prompt string from a list of class names."""
    class_list_str = ", ".join(class_names)
    if k_values is not None:
        prompt_template = prompt_template.format(k_values=k_values, class_list=class_list_str)
    else:
        prompt_template = prompt_template.format(class_list=class_list_str)
    return prompt_template

def batch_predict(model, tokenizer, all_images, class_names, prompt_template, device, batch_size=8, dtype=torch.bfloat16, k_values=None):
    """
    Performs prediction on a set of images given a list of candidate classes.
    This function processes images in batches for efficiency.
    """
    results = []
    generation_config = dict(
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    question = create_prompt(class_names, prompt_template, k_values)
    
    print(f"Preparing {len(all_images)} images for prediction...")
    print(f"Choosing from {len(class_names)} candidate classes.")

    # Start timing for prediction
    start_time = time.time()

    with torch.no_grad():
        # Create batches
        for i in tqdm(range(0, len(all_images), batch_size), desc="Running Batched Inference"):
            batch_images = all_images[i:i+batch_size]
            
            # Preprocess the batch of PIL images
            batch_pixel_values = []
            for image in batch_images:
                # load_image preprocesses one image and creates patches
                pixel_values = load_image(image, max_num=12).to(dtype).to(device)
                batch_pixel_values.append(pixel_values)

            # Since the number of patches can vary, we run inference image by image within the batch
            # A more advanced implementation could pad patches to run the model on the whole batch at once
            for j, pixel_values in enumerate(batch_pixel_values):
                try:
                    response = model.chat(tokenizer, pixel_values, question, generation_config)
                    results.append(response)
                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA Out of Memory on image {i+j+1}. Skipping.")
                    results.append("Prediction failed: Out of Memory")
                except Exception as e:
                    print(f"Error predicting image {i+j+1}: {e}")
                    results.append(f"Prediction failed: {str(e)}")
    
    # End timing for prediction
    end_time = time.time()
    prediction_time = end_time - start_time
    
    return results, prediction_time


def collect_data(dataloader, num_samples):
    """Collects a specified number of samples from the dataloader."""
    images, labels = [], []
    
    pbar = tqdm(total=num_samples, desc="Collecting data")
    
    collected_count = 0
    for images_batch, labels_batch in dataloader:
        if not images_batch: # Handles empty batches from safe_pil_collate
            continue

        if num_samples is not None and collected_count >= num_samples:
            break

        num_to_take = len(images_batch)
        if num_samples is not None:
            remaining = num_samples - collected_count
            if num_to_take > remaining:
                num_to_take = remaining
                images_batch = images_batch[:num_to_take]
                labels_batch = labels_batch[:num_to_take]

        images.extend(images_batch)
        labels.extend(labels_batch)
        
        collected_count += num_to_take
        pbar.update(num_to_take)

    pbar.close()

    if not images:
        print("No data collected from the dataloader.")
        return None, None

    # Transpose labels from [[c1, l1, s1], [c2, l2, s2]] to [[c1, c2], [l1, l2], [s1, s2]]
    labels_transposed = list(zip(*labels))
    
    # Create tensors for each label type
    corruption_labels = torch.tensor(labels_transposed[0])
    class_labels = torch.tensor(labels_transposed[1])
    severity_labels = torch.tensor(labels_transposed[2])

    return images, (corruption_labels, class_labels, severity_labels)


def evaluate(predictions, true_labels, class_names_map, task_name, quick_validate=False):
    """Evaluates the predictions and prints the results."""
    print(f"\n--- {task_name} Prediction Results ---")
    
    # For classification, class_names_map is a list. For corruption, it's a dict.
    if isinstance(class_names_map, dict):
        idx_to_name = {v: k for k, v in class_names_map.items()}
    else: # list
        idx_to_name = class_names_map

    # Display sample results
    for i in range(min(10, len(predictions))):
        true_idx = true_labels[i].item()
        true_name = idx_to_name.get(true_idx, "Unknown") if isinstance(idx_to_name, dict) else idx_to_name[true_idx]
        
        raw_prediction = predictions[i]
        predicted_name = raw_prediction.strip()
        is_correct = true_name.replace(" ", "").lower() == predicted_name.replace(" ", "").lower()
        
        print(f"\nImage {i+1}:")
        if quick_validate:
            print(f"  Raw Output: '{raw_prediction}'")
        print(f"  True Label: {true_name} (Index: {true_idx})")
        print(f"  Predicted Label: {predicted_name}")
        print(f"  Correct: {'Yes' if is_correct else 'No'}")

    # Calculate overall accuracy
    total_correct = 0
    for i in range(len(predictions)):
        true_idx = true_labels[i].item()
        true_name = idx_to_name.get(true_idx, "Unknown") if isinstance(idx_to_name, dict) else idx_to_name[true_idx]
        predicted_name = predictions[i].strip()
        if true_name.replace(" ", "").lower() == predicted_name.replace(" ", "").lower():
            total_correct += 1
            
    accuracy = (total_correct / len(predictions)) * 100 if len(predictions) > 0 else 0
    print(f"\nOverall {task_name} Accuracy: {total_correct}/{len(predictions)} = {accuracy:.2f}%")
    return accuracy


def evaluate_top_k(predictions, true_labels, class_names_map, task_name, quick_validate=False, k_values=3):
    """Evaluates top-k accuracy for comma-separated class name predictions."""
    print(f"\n--- {task_name} Top-K Prediction Results ---")
    print(f"Evaluating top-{k_values} corruption types.")
    
    # For classification, class_names_map is a list. For corruption, it's a dict.
    if isinstance(class_names_map, dict):
        idx_to_name = {v: k for k, v in class_names_map.items()}
        num_classes = len(class_names_map)
    else: # list
        idx_to_name = class_names_map
        num_classes = len(class_names_map)
    
    if k_values > num_classes:
        print(f"Skipping k={k_values} as it exceeds number of classes ({num_classes})")
        return {}
    
    accuracies = {}
    
    total_correct = 0
    for i in range(len(predictions)):
        true_idx = true_labels[i].item()
        true_name = idx_to_name.get(true_idx, "Unknown") if isinstance(idx_to_name, dict) else idx_to_name[true_idx]
        
        raw_prediction = predictions[i]
        
        # Parse comma-separated class names and clean them
        if isinstance(raw_prediction, str):
            # Remove trailing periods and extra whitespace, then split by comma
            cleaned_prediction = raw_prediction.rstrip('.').strip()
            predicted_names = [name.strip().rstrip('.') for name in cleaned_prediction.split(',')]
        else:
            predicted_names = [str(raw_prediction).strip().rstrip('.')]
        
        # Convert predicted names to indices
        predicted_indices = []
        for name in predicted_names:
            # Skip empty names
            if not name:
                continue
                
            # Find matching index for this name
            for idx, class_name in idx_to_name.items():
                if isinstance(idx_to_name, dict):
                    if class_name.replace(" ", "").lower() == name.replace(" ", "").lower():
                        predicted_indices.append(idx)
                        break
                else:
                    if idx_to_name[idx].replace(" ", "").lower() == name.replace(" ", "").lower():
                        predicted_indices.append(idx)
                        break
        
        # Take top-k predictions
        top_k_indices = predicted_indices[:k_values]
        
        # Check if true label is in top-k
        if true_idx in top_k_indices:
            total_correct += 1
        
        # Display sample results for first few images
        if i < 10 and quick_validate:
            top_k_names = []
            for idx in top_k_indices:
                name = idx_to_name.get(idx, "Unknown") if isinstance(idx_to_name, dict) else idx_to_name[idx]
                top_k_names.append(name)
            
            print(f"\nImage {i+1}:")
            print(f"  True Label: {true_name} (Index: {true_idx})")
            print(f"  Top-{k_values} Predictions: {', '.join(top_k_names)}")
            print(f"  Correct: {'Yes' if true_idx in top_k_indices else 'No'}")
    
    accuracy = (total_correct / len(predictions)) * 100 if len(predictions) > 0 else 0
    accuracies[f"top_{k_values}"] = round(accuracy, 2)
    print(f"\nTop-{k_values} Accuracy: {total_correct}/{len(predictions)} = {accuracy:.2f}%")
    
    return accuracies

def save_results_to_json(log_path, command, args, results, prompts, prediction_times):
    """Saves the experiment parameters and results to a JSON file."""
    
    results_dir = os.path.dirname(log_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "command": command,
        "parameters": vars(args),
        "results": results,
        "prompts": prompts,
        "prediction_times (seconds)": prediction_times
    }

    log_data = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing log file at {log_path}. Starting a new log.")
                log_data = []
    
    log_data.append(log_entry)

    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Results saved to {log_path}")


def main(args):
    """Main function to run the evaluation."""
    set_seed(args.seed)

    if args.quick_validate:
        print("\n--- Running in Quick Validation Mode ---")
        args.num_samples = 32
        args.batch_size = 8
        print(f"Number of samples set to {args.num_samples}")
        print(f"Batch size set to {args.batch_size}")
        print("----------------------------------------\n")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine optimal dtype
    if device.type == 'cuda':
        dtype = torch.bfloat16
    elif device.type == 'mps':
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"Using dtype: {dtype}")

    # Load model and tokenizer from Hugging Face
    print(f"Loading model and tokenizer from: {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    print("Model and tokenizer loaded successfully.")

    # Setup dataset and dataloader
    # We load PIL images directly now, so no transform is needed here.
    # The transform will be applied inside `load_image` on the patches.
    
    # Metadata files are expected to be in a fixed location within the repo
    metadata_source_path = "./data"
    print(f"Loading metadata from default path: {metadata_source_path}")

    if os.path.isdir(args.dataset_path):
        print(f"Loading dataset from folder: {args.dataset_path}")
        dataset = TTALoRADataset(
            dataset_folder=args.dataset_path,
            download_dataset=False,
            target_model='MLLM',
            transform=None, # No transform here, PIL images will be loaded
            quick_validate=args.quick_validate
        )
    else:
        print(f"Loading dataset from webdataset source: {args.dataset_path}")
        dataset = WebTTALoRADataset(
            url=args.dataset_path,
            metadata_path=metadata_source_path,
            transform=None, # No transform here
            quick_validate=args.quick_validate
        )
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(dataset, WebTTALoRADataset), # Disable shuffle for IterableDataset
        collate_fn=safe_pil_collate,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    # Load class and corruption indices
    class_index_path = os.path.join(metadata_source_path, 'imagenet_class_index.json')
    corruption_index_path = os.path.join(metadata_source_path, 'corruption_index.json')

    with open(class_index_path, 'r') as f:
        class_index = json.load(f)
    class_names = [details[1] for details in class_index.values()]

    with open(corruption_index_path, 'r') as f:
        corruption_index = json.load(f)
    corruption_names = list(corruption_index.keys())

    # Determine the number of samples to process
    num_samples_to_collect = args.num_samples
    if num_samples_to_collect is None:
        try:
            # For map-style datasets (folders), we can get the full length
            num_samples_to_collect = len(dataset)
            print(f"No --num_samples provided. Evaluating all {num_samples_to_collect} samples from the dataset.")
        except TypeError:
            # For iterable datasets (webdataset), length is unknown beforehand
            print("No --num_samples provided. Evaluating all samples from the iterable dataset.")

    # Collect data
    images, labels = collect_data(dataloader, num_samples_to_collect)
    if images is None:
        print("Could not collect any data. Exiting.")
        return

    print(f"\nCollected {len(images)} images for evaluation.")

    # --- Task: Classification ---
    results_dict = {}
    prompts_used = {}
    prediction_times = {"classification_time": None, "corruption_time": None}
    
    if args.task in ['classification', 'all']:
        # prompt_template = (
        #     "<image>\n"
        #     "Please select the most appropriate word from the following 1000 ImageNet categories that best describes the content of the image: [{class_list}]. "
        #     "Please provide only one word as your answer, ensure it is from the provided list, and do not add any explanation."
        # )
        prompt_template = (
            "<image>\n"
            "As an expert image analyst, identify the primary category of the subject in the image.\n\n"
            "Choose your answer from the following list: [{class_list}]\n\n"
            "Your answer must be a single word from the list."
        )
        prompts_used["classification_prompt"] = prompt_template
        predictions, classification_time = batch_predict(model, tokenizer, images, class_names, prompt_template, device, batch_size=args.batch_size, dtype=dtype)
        classification_accuracy = evaluate(predictions, labels[1], class_names, "Classification", args.quick_validate)
        results_dict["classification_accuracy"] = round(classification_accuracy, 2)
        prediction_times["classification_time"] = round(classification_time, 2)

    # --- Task: Corruption Detection ---
    if args.task in ['corruption', 'all']:
        # prompt_template = (
        #     "<image>\n"
        #     "Please select the corruption type that best describes the image from the following list: [{class_list}]. "
        #     "Please provide only the name of the corruption type, ensure it is from the provided list, and do not add any explanation."
        # )
        if args.top_k_values:
            prompt_template = (
                "<image>\n"
                "As an expert in digital image forensics, identify the type of corruption present in this image.\n\n"
                "return top [{k_values}] corruption types in the image: [{class_list}]\n\n"
                "Your answer must be [{k_values}] words from the list separated by commas."
            )
            prompts_used["corruption_prompt"] = prompt_template
            predictions, corruption_time = batch_predict(model, tokenizer, images, corruption_names, prompt_template, device, batch_size=args.batch_size, dtype=dtype, k_values=args.top_k_values)
            # print(f"Predictions: {predictions}")
            corruption_accuracies = evaluate_top_k(predictions, labels[0], corruption_index, "Corruption Type", args.quick_validate, args.top_k_values)
            results_dict["corruption_type_accuracies"] = corruption_accuracies
            prediction_times["corruption_time"] = round(corruption_time, 2)
        else:
            prompt_template = (
                "<image>\n"
                "As an expert in digital image forensics, identify the type of corruption present in this image.\n\n"
                "Choose your answer from the following list: [{class_list}]\n\n"
                "Your answer must be a single word from the list."
            )
            prompts_used["corruption_prompt"] = prompt_template
            predictions, corruption_time = batch_predict(model, tokenizer, images, corruption_names, prompt_template, device, batch_size=args.batch_size, dtype=dtype)
            corruption_accuracy = evaluate(predictions, labels[0], corruption_index, "Corruption Type", args.quick_validate)
            results_dict["corruption_type_accuracy"] = round(corruption_accuracy, 2)
            prediction_times["corruption_time"] = round(corruption_time, 2)


    # --- Save results ---
    # Generate a descriptive log file name based on model and dataset
    model_name_short = args.model_name.split('/')[-1]
    
    dataset_base = os.path.basename(args.dataset_path).lower()
    if 'nano-imagenet-c' in dataset_base:
        dataset_name_short = 'nano'
    elif 'mini-imagenet-c' in dataset_base:
        dataset_name_short = 'mini'
    elif 'imagenet-c' in dataset_base:
        dataset_name_short = 'full'
    else:
        dataset_name_short = dataset_base.replace('.tar', '')

    log_filename = f"{model_name_short}_on_{dataset_name_short}_log.json"
    log_path = os.path.join("./results", log_filename)

    command = "python " + " ".join(sys.argv)
    save_results_to_json(log_path, command, args, results_dict, prompts_used, prediction_times)


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
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-1B", help="Hugging Face model identifier.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of images to evaluate. If not provided, evaluates the entire dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--task", type=str, default="all", choices=['classification', 'corruption', 'all'], help="Task to perform.")
    parser.add_argument("--seed", type=int, default=7600, help="Random seed for reproducibility.")
    parser.add_argument("--quick_validate", action='store_true', help="Run in quick validation mode with a small subset of data.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for computation (e.g., 'cpu', 'cuda'). Defaults to auto-detection.")
    parser.add_argument("--top_k_values", type=int, default=None, help="Top-k values to evaluate. If not provided, evaluates only top-1 accuracy.")
    
    args = parser.parse_args()
    main(args)

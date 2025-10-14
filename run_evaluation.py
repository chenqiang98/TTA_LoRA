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

def create_prompt(class_names, prompt_template):
    """Creates a prompt string from a list of class names."""
    class_list_str = ", ".join(class_names)
    return prompt_template.format(class_list=class_list_str)

def batch_predict(model, tokenizer, images_batch, class_names, prompt_template):
    """
    Performs batch prediction on a set of images given a list of candidate classes.
    """
    results = []
    generation_config = dict(
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    question = create_prompt(class_names, prompt_template)

    all_pixel_values = []
    print(f"Preparing {images_batch.shape[0]} images for prediction...")
    print(f"Choosing from {len(class_names)} candidate classes.")

    for i in tqdm(range(images_batch.shape[0]), desc="Preprocessing images"):
        single_image = images_batch[i]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        denorm_image = torch.clamp(single_image * std + mean, 0, 1)
        pil_image = T.ToPILImage()(denorm_image)
        pixel_values = load_image(pil_image, max_num=12).to(torch.bfloat16).cuda()
        all_pixel_values.append(pixel_values)

    print("Starting batch inference...")
    with torch.no_grad():
        batch_size = len(all_pixel_values)
        for i in tqdm(range(batch_size), desc="Running Inference"):
            try:
                response = model.chat(tokenizer, all_pixel_values[i], question, generation_config)
                results.append(response)
            except Exception as e:
                print(f"Error predicting image {i+1}: {e}")
                results.append(f"Prediction failed: {str(e)}")
    return results

def collect_data(dataloader, num_samples):
    """Collects a specified number of samples from the dataloader."""
    images, labels = [], ([], [], [])
    
    # Initialize tqdm with total=num_samples. It correctly handles None for unknown totals.
    pbar = tqdm(total=num_samples, desc="Collecting data")
    
    collected_count = 0
    for images_batch, labels_batch in dataloader:
        if num_samples is not None and collected_count >= num_samples:
            break

        num_to_take = len(images_batch)
        if num_samples is not None:
            remaining = num_samples - collected_count
            if num_to_take > remaining:
                num_to_take = remaining
                images_batch = images_batch[:num_to_take]
                labels_batch = [l[:num_to_take] for l in labels_batch]

        images.append(images_batch)
        for i in range(len(labels)):
            labels[i].append(labels_batch[i])
        
        collected_count += num_to_take
        pbar.update(num_to_take)

    pbar.close()

    if not images:
        print("No data collected from the dataloader.")
        return None, None

    combined_images = torch.cat(all_images, dim=0)[:num_samples]
    combined_labels = [torch.cat([label[j] for label in all_labels], dim=0)[:num_samples] for j in range(len(all_labels[0]))]
    return combined_images, combined_labels

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

def save_results_to_json(log_path, command, args, results):
    """Saves the experiment parameters and results to a JSON file."""
    
    results_dir = os.path.dirname(log_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "command": command,
        "parameters": vars(args),
        "results": results
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

    # Setup dataset and dataloader
    mllm_transform = build_transform(224)
    
    # Metadata files are expected to be in a fixed location within the repo
    metadata_source_path = "./data"
    print(f"Loading metadata from default path: {metadata_source_path}")

    if os.path.isdir(args.dataset_path):
        print(f"Loading dataset from folder: {args.dataset_path}")
        dataset = TTALoRADataset(
            dataset_folder=args.dataset_path,
            download_dataset=False,
            target_model='MLLM',
            transform=mllm_transform,
            quick_validate=args.quick_validate
        )
    else:
        print(f"Loading dataset from webdataset source: {args.dataset_path}")
        dataset = WebTTALoRADataset(
            url=args.dataset_path,
            metadata_path=metadata_source_path,
            transform=mllm_transform,
            quick_validate=args.quick_validate
        )
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(dataset, WebTTALoRADataset), # Disable shuffle for IterableDataset
        worker_init_fn=worker_init_fn
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

    print(f"\nCollected {images.shape[0]} images for evaluation.")

    # --- Task: Classification ---
    results_dict = {}
    if args.task in ['classification', 'all']:
        prompt_template = (
            "<image>\n"
            "Please select the most appropriate word from the following 1000 ImageNet categories that best describes the content of the image: [{class_list}]. "
            "Please provide only one word as your answer, ensure it is from the provided list, and do not add any explanation."
        )
        predictions = batch_predict(model, tokenizer, images, class_names, prompt_template)
        classification_accuracy = evaluate(predictions, labels[1], class_names, "Classification", args.quick_validate)
        results_dict["classification_accuracy"] = round(classification_accuracy, 2)

    # --- Task: Corruption Detection ---
    if args.task in ['corruption', 'all']:
        prompt_template = (
            "<image>\n"
            "Please select the corruption type that best describes the image from the following list: [{class_list}]. "
            "Please provide only the name of the corruption type, ensure it is from the provided list, and do not add any explanation."
        )
        predictions = batch_predict(model, tokenizer, images, corruption_names, prompt_template)
        corruption_accuracy = evaluate(predictions, labels[0], corruption_index, "Corruption Type", args.quick_validate)
        results_dict["corruption_type_accuracy"] = round(corruption_accuracy, 2)

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
    save_results_to_json(log_path, command, args, results_dict)


if __name__ == "__main__":
    """
    Example Usage:
    ------------------------------------------------------------------------------------
    All examples use the default model ('OpenGVLab/InternVL3_5-1B') for consistency, unless specified otherwise.
    
    1. Evaluate on the local 'nano-imagenet-c' folder dataset:
       python run_evaluation.py --dataset_path ./data/nano-imagenet-c
    
    2. Evaluate with a different model (e.g., SmolVLM) on the nano dataset:
       python run_evaluation.py \
           --dataset_path ./data/nano-imagenet-c \
           --model_name HuggingFaceTB/SmolVLM-500M-Instruct

    3. Evaluate on the 'nano-imagenet-c' dataset from Hugging Face Hub (evaluates all samples):
       python run_evaluation.py --dataset_path niuniandaji/nano-imagenet-c
    
    4. Evaluate on a local .tar webdataset file, limiting to 100 samples for a quick test:
       python run_evaluation.py \
           --dataset_path ./data/nano-ImageNet-C.tar \
           --num_samples 100

    5. Run in quick validation mode (uses a small, fixed number of samples and smaller batch size):
       python run_evaluation.py --quick_validate
    
    6. Evaluate only the 'corruption' detection task on the nano dataset folder:
       python run_evaluation.py \
           --dataset_path ./data/nano-imagenet-c \
           --task corruption
    ------------------------------------------------------------------------------------
    
    Notes on available models and datasets:
    
    - **Models**: This script is compatible with various Vision-Language Models from Hugging Face.
      Some tested models include:
      - OpenGVLab/InternVL3_1B
      - OpenGVLab/InternVL3_5-1B
      - OpenGVLab/InternVL3_5-2B
      - HuggingFaceTB/SmolVLM-256M-Instruct
      - HuggingFaceTB/SmolVLM-500M-Instruct

    - **Datasets**: The --dataset_path argument supports three formats:
      1. Path to a local folder (e.g., ./data/mini-ImageNet-C)
      2. Path to a local .tar webdataset file (e.g., ./data/nano-ImageNet-C.tar)
      3. A Hugging Face Hub dataset ID (e.g., niuniandaji/nano-imagenet-c)
    """
    
    parser = argparse.ArgumentParser(description="Evaluate Vision-Language Models on ImageNet-C")
    parser.add_argument("--dataset_path", type=str, default="./data/nano-imagenet-c", help="Path to the dataset (folder, .tar file, or HF repo ID).")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-1B", help="Hugging Face model identifier.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of images to evaluate. If not provided, evaluates the entire dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the DataLoader.")
    parser.add_argument("--task", type=str, default="all", choices=['classification', 'corruption', 'all'], help="Task to perform.")
    parser.add_argument("--seed", type=int, default=7600, help="Random seed for reproducibility.")
    parser.add_argument("--quick_validate", action='store_true', help="Run in quick validation mode with a small subset of data.")
    
    args = parser.parse_args()
    main(args)

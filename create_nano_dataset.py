import argparse
import os
import random
import shutil
from tqdm import tqdm
import webdataset as wds
from huggingface_hub import HfApi, HfFolder

def sample_and_copy_files(source_dir, output_dir, num_samples, seed, scan_report_interval):
    """
    Scans a source directory, randomly samples image files, and copies them 
    to an output directory, preserving the original structure.

    IMPORTANT: This script is currently hardcoded to only sample images from
    corruption severity level 5.
    """
    print(f"1. Scanning for image files in '{source_dir}'...")
    all_files = []
    count = 0
    for root, _, files in os.walk(source_dir):
        # HARDCODED FILTER: Only include files from severity level 5 directories.
        # We check if '5' is a component of the directory path.
        path_parts = root.split(os.sep)
        if '5' in path_parts:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    all_files.append(os.path.join(root, file))
                    count += 1
                    if count > 0 and count % scan_report_interval == 0:
                        print(f"   ...scanned {count} images (from severity 5)...")
    
    if not all_files:
        print(f"Error: No image files found for severity level 5 in '{source_dir}'. Exiting.")
        return False
    
    print(f"Found {len(all_files)} total images.")

    # Sort the file list to ensure reproducibility across different systems.
    # The order of files returned by os.walk is not guaranteed to be the same.
    all_files.sort()

    print(f"2. Randomly sampling {num_samples} images (seed={seed})...")
    # Set the random seed for reproducibility.
    # Note: For 100% identical results, the same version of Python should be used
    # (e.g., this script was created using Python 3.10.14), as the underlying
    # algorithm for the 'random' module can change between versions.
    random.seed(seed)
    num_to_sample = min(num_samples, len(all_files))
    sampled_files = random.sample(all_files, num_to_sample)
    print(f"Selected {len(sampled_files)} files to copy.")

    print(f"3. Copying files to '{output_dir}'...")
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Removing it.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for file_path in tqdm(sampled_files, desc="Copying files"):
        relative_path = os.path.relpath(file_path, source_dir)
        dest_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)
    
    print("File copying complete.")
    return True

def generate_readme_content(repo_id, seed, python_version, tar_filename, script_filename):
    """
    Generates the content for the README.md file for the Hugging Face dataset.
    """
    return f"""---
license: mit
tags:
- image-classification
- computer-vision
- imagenet-c
---

# Nano ImageNet-C (Severity 5)

This is a randomly sampled subset of the ImageNet-C dataset, containing 5,000 images exclusively from corruption **severity level 5**. It is designed for efficient testing and validation of model robustness.

这是一个从 ImageNet-C 数据集中随机抽样的子集，包含 5000 张仅来自损坏等级为 **5** 的图像。它旨在用于高效地测试和验证模型的鲁棒性。

## How to Generate / 如何生成

This dataset was generated using the `{script_filename}` script included in this repository. To ensure reproducibility, the following parameters were used:

本数据集使用此仓库中包含的 `{script_filename}` 脚本生成。为确保可复现性，生成时使用了以下参数：

- **Source Dataset / 源数据集**: The full ImageNet-C dataset is required. / 需要完整的 ImageNet-C 数据集。
- **Random Seed / 随机种子**: `{seed}`
- **Python Version / Python 版本**: `{python_version}`

## Dataset Structure / 数据集结构

The dataset is provided as a single `.tar` file named `{tar_filename}` in the `webdataset` format. The internal structure preserves the original ImageNet-C hierarchy: `corruption_type/class_name/image.jpg`.

数据集以 `webdataset` 格式打包在名为 `{tar_filename}` 的单个 `.tar` 文件中。其内部结构保留了原始 ImageNet-C 的层次结构：`corruption_type/class_name/image.jpg`。

## Citation / 引用

If you use this dataset, please cite the original ImageNet-C paper:

如果您使用此数据集，请引用原始 ImageNet-C 的论文：

```bibtex
@inproceedings{{danhendrycks2019robustness,
  title={{Benchmarking Neural Network Robustness to Common Corruptions and Perturbations}},
  author={{Dan Hendrycks and Thomas Dietterich}},
  booktitle={{International Conference on Learning Representations}},
  year={{2019}},
  url={{https://openreview.net/forum?id=HJz6tiCqYm}},
}}
```
"""

def package_with_webdataset(output_dir, tar_path):
    """
    Packages the contents of a directory into a single .tar file using webdataset.
    """
    print(f"4. Packaging '{output_dir}' into '{tar_path}'...")
    
    with wds.TarWriter(tar_path) as sink:
        for root, _, files in tqdm(list(os.walk(output_dir)), desc="Packaging files"):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as stream:
                    content = stream.read()
                
                relative_path = os.path.relpath(file_path, output_dir)
                key, ext = os.path.splitext(relative_path)
                extension = ext.lstrip('.')
                
                sink.write({
                    "__key__": key,
                    extension: content
                })
    
    print("Packaging complete.")

def upload_to_hf(tar_path, readme_path, script_path, repo_id):
    """
    Uploads a file to a specified Hugging Face Hub repository.
    """
    print(f"5. Uploading files to Hugging Face Hub repository: {repo_id}...")
    
    if HfFolder.get_token() is None:
        print("Hugging Face token not found. Please log in using `huggingface-cli login` first.")
        return

    try:
        api = HfApi()
        print(f"Creating repository '{repo_id}' (if it doesn't exist)...")
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        script_filename = os.path.basename(script_path)
        print(f"Uploading generation script '{script_filename}'...")
        api.upload_file(
            path_or_fileobj=script_path,
            path_in_repo=script_filename,
            repo_id=repo_id,
            repo_type="dataset"
        )

        tar_filename = os.path.basename(tar_path)
        print(f"Uploading dataset file '{tar_filename}'...")
        api.upload_file(
            path_or_fileobj=tar_path,
            path_in_repo=tar_filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("Upload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

def main():
    """
    Main function to orchestrate the dataset creation, packaging, and upload process.
    """
    parser = argparse.ArgumentParser(description="Create, package, and upload a smaller version of an image dataset.")
    parser.add_argument("--source_dir", type=str, default="./data/ImageNet-C", help="Path to the source dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/nano-ImageNet-C", help="Path to save the new sampled dataset.")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of images to sample.")
    parser.add_argument("--seed", type=int, default=7600, help="Random seed for reproducibility.")
    parser.add_argument("--repo_id", type=str, default="niuniandaji/nano-imagenet-c", help="The Hugging Face Hub repository ID.")
    parser.add_argument("--tar_path", type=str, default="./data/nano-ImageNet-C.tar", help="Path to save the final webdataset archive.")
    parser.add_argument("--scan_report_interval", type=int, default=50000, help="How often to report progress during file scanning.")
    args = parser.parse_args()

    print("--- Starting Dataset Creation Process ---")
    print("IMPORTANT: The script is configured to sample ONLY from severity level 5.")
    
    # Generate README content
    script_filename = os.path.basename(__file__)
    readme_content = generate_readme_content(
        args.repo_id, 
        args.seed, 
        "3.10.14", 
        os.path.basename(args.tar_path), 
        script_filename
    )
    
    # Write README to a local file
    readme_path = "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Generated README.md for the dataset.")

    if sample_and_copy_files(args.source_dir, args.output_dir, args.num_samples, args.seed, args.scan_report_interval):
        package_with_webdataset(args.output_dir, args.tar_path)
        upload_to_hf(args.tar_path, readme_path, script_filename, args.repo_id)

    print("--- Process Finished ---")

if __name__ == "__main__":
    main()

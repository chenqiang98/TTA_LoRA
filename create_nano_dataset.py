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
    """
    print(f"1. Scanning for image files in '{source_dir}'...")
    all_files = []
    count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_files.append(os.path.join(root, file))
                count += 1
                if count > 0 and count % scan_report_interval == 0:
                    print(f"   ...scanned {count} images...")
    
    if not all_files:
        print(f"Error: No image files found in '{source_dir}'. Exiting.")
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

def upload_to_hf(tar_path, repo_id):
    """
    Uploads a file to a specified Hugging Face Hub repository.
    """
    print(f"5. Uploading '{tar_path}' to Hugging Face Hub repository: {repo_id}...")
    
    if HfFolder.get_token() is None:
        print("Hugging Face token not found. Please log in using `huggingface-cli login` first.")
        return

    try:
        api = HfApi()
        print(f"Creating repository '{repo_id}' (if it doesn't exist)...")
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        
        print("Uploading file...")
        api.upload_file(
            path_or_fileobj=tar_path,
            path_in_repo=os.path.basename(tar_path),
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
    
    if sample_and_copy_files(args.source_dir, args.output_dir, args.num_samples, args.seed, args.scan_report_interval):
        package_with_webdataset(args.output_dir, args.tar_path)
        upload_to_hf(args.tar_path, args.repo_id)

    print("--- Process Finished ---")

if __name__ == "__main__":
    main()

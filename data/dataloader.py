import os
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import tarfile
import io
from tqdm import tqdm
import webdataset as wds
from torch.utils.data import IterableDataset
import numpy as np

image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def worker_init_fn(worker_id):
    """
    Sets the seed for each worker in a DataLoader.
    """
    seed = 7600
    np.random.seed(seed)

class WebTTALoRADataset(IterableDataset):
    """
    A dataset class for loading TTALoRA data from a webdataset.
    This is an IterableDataset, suitable for streaming data.
    """
    def __init__(self, url, metadata_path, transform=None, quick_validate=False):
        """
        Args:
            url (string): URL or path to the webdataset .tar file(s).
            metadata_path (string): Path to the directory containing class_index.json and corruption_index.json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.url = url
        self.transform = transform
        self.quick_validate = quick_validate

        # Load metadata
        self.class_index_path = os.path.join(metadata_path, 'class_index.json')
        self.corruption_index_path = os.path.join(metadata_path, 'corruption_index.json')
        
        with open(self.class_index_path, 'r') as f:
            self.class_index = json.load(f)
        with open(self.corruption_index_path, 'r') as f:
            self.corruption_index = json.load(f)
        
        # Create a reverse mapping from class name (e.g., n01440764) to class index (e.g., 0)
        self.class_name_to_idx = {details[0]: int(idx) for idx, details in self.class_index.items()}

    def __iter__(self):
        # Create the dataset pipeline
        # If the URL is from Hugging Face, prepend with 'hf://'
        if "hf.co" in self.url or self.url.count("/") == 1:
             dataset_url = f"pipe:curl -L hf://datasets/{self.url}/resolve/main/{os.path.basename(self.url).replace('.tar','')}.tar"
        else:
             dataset_url = self.url
        
        dataset = wds.WebDataset(dataset_url).decode("pil")

        for sample in dataset:
            # Extract key and image
            key = sample['__key__']
            image = sample['png'] if 'png' in sample else sample['jpg']

            # Parse the key to get labels
            try:
                corruption_name, severity_str, class_name, _ = key.split('/')
                severity = int(severity_str)
                
                corruption_label = self.corruption_index[corruption_name]
                class_label = self.class_name_to_idx[class_name]

                if self.transform:
                    image = self.transform(image)
                
                yield image, (corruption_label, class_label, severity)

            except (ValueError, KeyError) as e:
                # This can happen if the key format is unexpected.
                # print(f"Skipping sample with malformed key '{key}': {e}")
                continue


class TTALoRADataset(torch.utils.data.Dataset):
    """
    A custom dataset for TTA-LoRA evaluation on Mini-ImageNet-C.
    """
    def __init__(self, dataset_folder, download_dataset=False, target_model='ViT', transform=None, quick_validate=False):
        """
        Initializes the dataset.
        Args:
            dataset_folder (str): The path to the dataset folder.
            download_dataset (bool): Whether to download the dataset if not found.
            target_model (str): The target model type ('ViT' or 'MLLM').
            transform (callable, optional): Optional transform to be applied on a sample.
            quick_validate (bool): If True, loads only a small subset of data for quick validation.
        """
        self.data_process_function = None
        self.dataset_folder = Path(dataset_folder)
        self.target_model = target_model
        self.transform = transform
        self.quick_validate = quick_validate

        # 设置默认的文件路径，基于当前文件的位置
        current_dir = Path(__file__).parent
        self.corruption_index = current_dir / 'corruption_index.json'
        self.class_index = current_dir / 'imagenet_class_index.json'
        
        if not self.dataset_folder.exists() and download_dataset:
            self._download_and_extract_dataset()

        self.image_paths, self.labels = self._load_image_paths_and_labels(
            self.dataset_folder, self.class_index, self.corruption_index, self.quick_validate
        )

    def _download_dataset(self):
        raise NotImplementedError(f"Download dataset is not implemented, please go to {self.dataset_url} to download the dataset")

    def _load_image_paths_and_labels(self, dataset_folder, class_index_path, corruption_index_path, quick_validate=False):
        """
        Loads image paths and labels from the dataset folder.
        Supports both tarball and regular folder structures.
        """
        image_paths = []
        labels = []

        with open(class_index_path, 'r') as f:
            class_index_data = json.load(f)

        with open(corruption_index_path, 'r') as f:
            corruption_index_data = json.load(f)
        
        corruption_types = list(corruption_index_data.keys())
        if quick_validate:
            corruption_types = corruption_types[:1]
            print(f"\n[Quick Validation Mode] Loading only the first corruption type: {corruption_types[0]}")

        for corruption_type in tqdm(corruption_types, desc="Loading dataset"):
            corruption_path = dataset_folder / corruption_type
            if not corruption_path.exists():
                print(f"Warning: Corruption directory not found: {corruption_path}")
                continue
                
            if corruption_type not in corruption_index_data:
                print(f"警告: corruption类型 '{corruption_type}' 不在索引文件中，跳过")
                continue
            
            corruption_type_data = corruption_index_data[corruption_type]
            
            # 遍历severity级别文件夹
            for severity_name in os.listdir(corruption_path):
                severity_path = corruption_path / severity_name
                if not severity_path.is_dir():
                    continue
                
                try:
                    severity = int(severity_name)
                except ValueError:
                    print(f"警告: 无法从目录名 {severity_name} 解析severity，跳过")
                    continue
                
                # 检查是否为tar文件格式的WebDataset
                tar_files = list(severity_path.glob("*.tar"))
                if tar_files:
                    # 处理WebDataset格式
                    for tar_file in tar_files:
                        try:
                            with tarfile.open(tar_file, 'r') as tar:
                                for member in tar.getmembers():
                                    if member.name.endswith('.jpg') or member.name.endswith('.jpeg') or member.name.endswith('.png'):
                                        # 从tar文件中提取图像路径信息
                                        base_name = os.path.splitext(member.name)[0]
                                        
                                        # 查找对应的类别标签文件
                                        cls_member_name = base_name + '.cls'
                                        cls_member = None
                                        for m in tar.getmembers():
                                            if m.name == cls_member_name:
                                                cls_member = m
                                                break
                                        
                                        if cls_member:
                                            # 提取类别标签
                                            cls_data = tar.extractfile(cls_member)
                                            if cls_data:
                                                try:
                                                    class_index = int(cls_data.read().decode('utf-8').strip())
                                                    # 存储tar文件路径和成员名称的元组
                                                    image_paths.append((str(tar_file), member.name))
                                                    labels.append([corruption_type_data, class_index, severity])
                                                except (ValueError, UnicodeDecodeError):
                                                    print(f"警告: 无法解析类别标签 {cls_member_name}")
                        except Exception as e:
                            print(f"警告: 处理tar文件 {tar_file} 时出错: {e}")
                else:
                    # 处理普通文件夹格式，其结构为 severity/class/image.jpg
                    # 首先，为了提高效率，创建一个从WordNet ID到类别索引的逆向映射
                    wordnet_to_index = {v[0]: int(k) for k, v in class_index_data.items()}

                    for class_dir in os.listdir(severity_path):
                        class_path = severity_path / class_dir
                        if not os.path.isdir(class_path):
                            continue
                        
                        # 目录名即为类别ID (例如 'n01440764')
                        class_id = class_dir
                        class_index = wordnet_to_index.get(class_id)
                        
                        if class_index is not None:
                            for image_file in os.listdir(class_path):
                                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    image_path = class_path / image_file
                                    image_paths.append(str(image_path))
                                    labels.append([corruption_type_data, class_index, severity])

            print(f"从{corruption_path}加载了 {len([p for p in image_paths if (isinstance(p, tuple) and str(corruption_path) in p[0]) or (isinstance(p, str) and str(corruption_path) in p)])} 张图片")
        
        return image_paths, labels
    
    def _extract_class_index_from_filename(self, filename, class_index_data):
        """
        从文件名中提取类别索引
        """
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 尝试从ImageNet类别索引中匹配
        for class_idx, (synset_id, class_name) in class_index_data.items():
            if synset_id in filename or class_name.lower() in name_without_ext.lower():
                return int(class_idx)
        
        # 如果无法从文件名推断，返回None
        print(f"警告: 无法从文件名 '{filename}' 中提取类别索引")
        return None

    def _load_image_from_tar(self, tar_path, member_name):
        """
        从tar文件中加载图像
        """
        try:
            with tarfile.open(tar_path, 'r') as tar:
                member = tar.getmember(member_name)
                image_data = tar.extractfile(member)
                if image_data:
                    image = Image.open(io.BytesIO(image_data.read())).convert('RGB')
                    return image
        except Exception as e:
            print(f"从tar文件加载图像时出错: {tar_path}#{member_name}, 错误: {e}")
        return None

    def _apply_transforms(self, sample):
        """
        对样本应用变换
        """
        image, label = sample
        
        # 如果有自定义的数据处理函数，先应用它
        if self.data_process_function is not None:
            image = self.data_process_function(image)
        
        # 应用图像变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __getitem__(self, index):
        """
        根据索引获取数据样本
        """
        # 加载图像
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        try:
            # 检查是否为tar文件格式
            if isinstance(image_path, tuple):
                # tar文件格式: (tar_path, member_name)
                tar_path, member_name = image_path
                image = self._load_image_from_tar(tar_path, member_name)
                if image is None:
                    # Error is printed in the helper function, return None to skip.
                    return None, None
            else:
                # 普通文件路径
                image = Image.open(image_path).convert('RGB')
            
            # Apply transforms if they are provided
            image, label = self._apply_transforms((image, label))
            
            return image, label
            
        except Exception as e:
            # This will catch errors from Image.open() for corrupted files
            print(f"Warning: Could not load image {image_path}: {e}. Skipping.")
            return None, None

    def __iter__(self):
        """
        返回数据集的迭代器
        """
        for i in range(len(self.image_paths)):
            yield self.__getitem__(i)

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.image_paths)

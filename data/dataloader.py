import os
import json
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import tarfile
import io

image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class TTALoRADataset(torch.utils.data.Dataset):
    def __init__(self, 
               dataset_folder = None,
               download_dataset = False, # 是否下载数据集
               target_model = None, 
               corruption_index = None, 
               class_index = None,
                transform = image_transform,
                input_size=224, 
                max_num=12, 
                data_process_function = None, # 每一个图片在调用前的处理函数，输入为一张图片，留空为不处理
                kwargs = None, # 其他参数
                ) -> None:
        self.dataset_folder = dataset_folder
        self.target_model = target_model
        
        # 设置默认的文件路径，基于当前文件的位置
        current_dir = Path(__file__).parent
        self.corruption_index = corruption_index if corruption_index else current_dir / 'corruption_index.json'
        self.class_index = class_index if class_index else current_dir / 'imagenet_class_index.json'
        
        self.transform = transform
        self.input_size = input_size
        self.max_num = max_num
        self.data_process_function = data_process_function
        self.kwargs = kwargs

        self.dataset_url = 'https://huggingface.co/datasets/niuniandaji/mini-imagenet-c'

        if download_dataset:
            self._download_dataset()

        # 初始化时加载数据集
        self.image_paths, self.labels = self._load_dataset()

    def _download_dataset(self):
        raise NotImplementedError(f"Download dataset is not implemented, please go to {self.dataset_url} to download the dataset")

    def _load_dataset(self):
        """
        加载数据集，返回包含图像路径和标签的列表
        标签格式为 [corruption_type, class_index]
        """
        if not self.dataset_folder:
            raise ValueError("数据集文件夹路径不能为空")
        
        dataset_root = Path(self.dataset_folder)
        if not dataset_root.exists():
            raise FileNotFoundError(f"数据集文件夹不存在: {self.dataset_folder}")
        
        # 加载类别索引映射
        with open(self.class_index, 'r') as f:
            class_index_data = json.load(f)
        
        # 加载corruption索引映射
        with open(self.corruption_index, 'r') as f:
            corruption_index_data = json.load(f)
        
        image_paths = []
        labels = []
        
        # 遍历corruption类型文件夹
        for corruption_name in os.listdir(dataset_root):
            corruption_path = dataset_root / corruption_name
            if not corruption_path.is_dir() or corruption_name == '.git':
                continue
                
            if corruption_name not in corruption_index_data:
                print(f"警告: corruption类型 '{corruption_name}' 不在索引文件中，跳过")
                continue
            
            corruption_type = corruption_index_data[corruption_name]
            
            # 遍历severity级别文件夹
            for severity_name in os.listdir(corruption_path):
                severity_path = corruption_path / severity_name
                if not severity_path.is_dir():
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
                                                    labels.append([corruption_type, class_index])
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
                                    labels.append([corruption_type, class_index])

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
                    raise ValueError(f"无法从tar文件加载图像: {tar_path}#{member_name}")
            else:
                # 普通文件路径
                image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            image, label = self._apply_transforms((image, label))
            
            return image, label
            
        except Exception as e:
            print(f"加载图像时出错: {image_path}, 错误: {e}")
            # 返回一个默认的空白图像和标签，避免中断训练
            default_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            default_image, label = self._apply_transforms((default_image, label))
            return default_image, label

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

import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch

# ---------- backbone ----------
def get_resnet(model_name='resnet50', pretrained=True):
    weight_map = {
        'resnet50': ResNet50_Weights.IMAGENET1K_V1,
        'resnet18': ResNet18_Weights.IMAGENET1K_V1,
    }
    weights = weight_map.get(model_name) if pretrained else None
    if not hasattr(models, model_name):
        raise ValueError(f"Model {model_name} not available")
    return getattr(models, model_name)(weights=weights)

def resnet50(pretrained=True):
    return get_resnet('resnet50', pretrained)

def resnet18(pretrained=True):
    return get_resnet('resnet18', pretrained)

# ---------- transforms ----------
# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_resnet_transform(input_size=224, is_training=True):
    """
    Get the standard transform for ResNet models.
    
    Args:
        input_size (int): Input image size (default: 224 for ResNet)
        is_training (bool): Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform

def get_resnet_eval_transform(input_size=224):
    """Get evaluation transform for ResNet models."""
    return get_resnet_transform(input_size, is_training=False)

def get_resnet_train_transform(input_size=224):
    """Get training transform for ResNet models."""
    return get_resnet_transform(input_size, is_training=True)

# Predefined transforms for common use cases
resnet_transform = get_resnet_eval_transform()
resnet_train_transform = get_resnet_train_transform()

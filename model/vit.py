import timm
from torchvision import transforms
from PIL import Image
import torch

def get_vit(model_name='vit_base_patch16_224', pretrained=True):
    # Use timm to load ViT-Base to avoid torchvision availability issues
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

def get_mobilevit(model_name='mobilevit_s', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

vit_base_patch16_224 = lambda pretrained=True: get_vit('vit_base_patch16_224', pretrained)
mobilevit_s = lambda pretrained=True: get_mobilevit('mobilevit_s', pretrained)

# ---------- transforms ----------
# ImageNet normalization values (same as ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_vit_transform(input_size=224, is_training=True):
    """
    Get the standard transform for ViT models.
    
    Args:
        input_size (int): Input image size (default: 224 for ViT-Base)
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

def get_vit_eval_transform(input_size=224):
    """Get evaluation transform for ViT models."""
    return get_vit_transform(input_size, is_training=False)

def get_vit_train_transform(input_size=224):
    """Get training transform for ViT models."""
    return get_vit_transform(input_size, is_training=True)

def get_mobilevit_transform(input_size=224, is_training=True):
    """
    Get the standard transform for MobileViT models.
    
    Args:
        input_size (int): Input image size (default: 224)
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

def get_mobilevit_eval_transform(input_size=224):
    """Get evaluation transform for MobileViT models."""
    return get_mobilevit_transform(input_size, is_training=False)

def get_mobilevit_train_transform(input_size=224):
    """Get training transform for MobileViT models."""
    return get_mobilevit_transform(input_size, is_training=True)

# Predefined transforms for common use cases
vit_transform = get_vit_eval_transform()
vit_train_transform = get_vit_train_transform()
mobilevit_transform = get_mobilevit_eval_transform()
mobilevit_train_transform = get_mobilevit_train_transform()
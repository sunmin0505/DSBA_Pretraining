from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split
from .augmentation import get_augmentation, get_normalize
import torch

_SPLIT_CACHE = None

def get_cifar10_dataset(split: str, aug_level: str = "default", val_ratio: float = 0.1):
    """
    CIFAR-10 train/val/test 데이터셋 반환

    Args:
        split (str): 'train', 'val', 'test'
        aug_level (str): augmentation 수준 ('default', 'weak', 'strong')
        val_ratio (float): train-validation 비율

    Returns:
        CIFAR10 또는 Subset 객체
    """
    assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"

    if split == 'test':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            get_normalize()
        ])
        return CIFAR10(root='./data', train=False, transform=transform, download=True)

    # train or val
    full_dataset = CIFAR10(
        root='./data',
        train=True,
        transform=get_augmentation(aug_level),
        download=True
    )

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    global _SPLIT_CACHE
    if _SPLIT_CACHE is None:
        generator = torch.Generator().manual_seed(42)
        _SPLIT_CACHE = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_set, val_set = _SPLIT_CACHE

    if split == 'train':
        return train_set
    else:
        # val에는 augmentation 제거
        val_set.dataset.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            get_normalize()
        ])
        return val_set
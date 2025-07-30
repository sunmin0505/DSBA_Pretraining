from torchvision.datasets import CIFAR10
from torchvision import transforms
from .augmentation import get_augmentation, get_normalize

def get_cifar10_dataset(split: str, aug_level: str = "default"):
    """
    CIFAR-10 train/test 데이터셋 반환

    Args:
        split (str): 'train' 또는 'test'
        aug_level (str): augmentation 수준 ('default', 'weak', 'strong')
                         test split에서는 무시됨

    Returns:
        torchvision.datasets.CIFAR10 객체
    """
    assert split in ['train', 'test'], "split must be 'train' or 'test'"

    is_train = (split == 'train')

    if is_train:
        transform = get_augmentation(aug_level)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            get_normalize()
        ])

    dataset = CIFAR10(
        root='./data',
        train=is_train,
        transform=transform,
        download=True
    )

    return dataset

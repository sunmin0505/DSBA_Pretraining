from torchvision import transforms

def get_normalize():
    """
    CIFAR-10 정규화 transform 리턴
    """
    _CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    _CIFAR10_STD = [0.2470, 0.2435, 0.2616]

    return transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)

def get_augmentation(level: str):
    """
    CIFAR-10 augmentation 구성

    Args:
        level (str): 'default', 'weak', 'strong'

    Returns:
        torchvision.transforms.Compose
    """
    normalize = get_normalize()

    if level == 'default':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(), # Image를 [0, 255] -> [0.0, 1.0] 범위의 float32 tensor로 변환 
            normalize
        ])

    elif level == 'weak':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=32), # 224x224 이미지에 32픽셀 패딩 -> 300x300 이미지 생성 -> 224x224 이미지로 랜덤하게 자름
            transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
            transforms.ToTensor(),
            normalize
        ])

    elif level == 'strong':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 원본 이미지의 80% ~ 100% 크기로 랜덤하게 자름 -> 다시 224x224로 resize
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), # 색상, 대비, 채도, 색조 변형형
            transforms.ToTensor(),
            normalize
        ])

    else:
        raise ValueError(f"Unsupported augmentation level: {level}")

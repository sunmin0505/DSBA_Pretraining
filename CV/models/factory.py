import timm
import torch.nn as nn

def create_model(model_name: str, num_classes: int):
    """
    timm 라이브러리를 이용해 모델 생성하고 classifier를 교체함

    Args:
        model_name (str): 모델 이름 (ex. 'resnet18', 'vit_base_patch16_224')
        num_classes (int): 출력 클래스 수

    Returns:
        nn.Module: 분류용 모델
    """
    try:
        model = timm.create_model(model_name, pretrained=True)
    except Exception as e:
        raise ValueError(f"[create_model] 지원하지 않는 모델입니다: {model_name}") from e

    # classifier 교체 (모델 구조에 따라 방식 다름)
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(num_classes)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"[create_model] 분류기 교체를 지원하지 않는 모델입니다: {model_name}")

    return model

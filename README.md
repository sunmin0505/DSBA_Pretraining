## DSBA_Study

# CV_Classification

실험 목적
- CIFAR-10 데이터셋 기반 ResNet, ViT 비교
- 각 모델별로 data augmentation(default, weak, strong) 수행

Data Augmentation
- default: Noramalization
- weak: RandomCrop + HorizontalFlip
- strong: RandomResizedCrop + HorizontalFlip + ColorJitter

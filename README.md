## DSBA_Study

# CV_Classification

실험 목적
- CIFAR-10 데이터셋 기반 ResNet, ViT 비교
- 각 모델별로 data augmentation(default, weak, strong) 수행

Data Augmentation
- DEFAULT: Noramalization
- WEAK: RandomCrop + HorizontalFlip
- STRONG: RandomResizedCrop + HorizontalFlip + ColorJitter

Model
- resnet50
- vit_small_patch16_224

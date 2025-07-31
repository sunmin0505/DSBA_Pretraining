# DSBA_Study

## CV_Classification

실험 목적
- CIFAR-10 데이터셋 기반 ResNet, ViT 비교
- 각 모델별로 data augmentation(default, weak, strong) 수행하여 그 결과를 비교

Data Augmentation
- DEFAULT: Noramalization
- WEAK:  Noramalization + RandomCrop(Padding) + HorizontalFlip
- STRONG: Normalization + RandomResizedCrop + HorizontalFlip + ColorJitter

Model (timm에서 불러옴. pre-trained된 모델)
- resnet50
- vit_small_patch16_224

가설
- 데이터셋이 작으니,, → CNN 기반 모델이 더 잘할 것이다?
- 적당한 증강이 성능 향상에 도움이 된다? (과적합 등의 문제,,)

결과
- https://wandb.ai/sunmin_kim-seoul-national-university/cifar10-aug-experiment?nw=nwusersunmin_kim

| Model | ACC |
| --- | --- |
| ResNet50 + DEFAULT | 0.9499 |
| ResNet50 + WEAK | 0.9515 |
| **ResNet50 + STRONG** | **0.9502** |
| ViT_small_patch16_224 + DEFAULT | 0.7863 |
| ViT_small_patch16_224 + WEAK | 0.8440 |
| ViT_small_patch16_224 + STRONG | 진행중 |

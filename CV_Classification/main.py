import torch
import wandb

from datasets import get_cifar10_dataset
from models import create_model
from train import train_one_epoch, evaluate
from config import configs

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in configs:
        for aug_level in ["default", "weak", "strong"]:
            config = configs[model_name]

            print("=" * 60)
            print(f"▶ 모델: {model_name}")
            print(f"▶ Augmentation: {aug_level}")
            print(f"▶ Epochs: {config['epochs']}, Batch Size: {config['batch_size']}, LR: {config['learning_rate']}")
            print("=" * 60)

            wandb.init(
                project="cifar10-aug-experiment",
                name=f"{model_name}_{aug_level}",
                config=config
            )

            # 모델 생성
            model = create_model(model_name, num_classes=config["num_classes"]).to(device)

            # 데이터셋 & 로더
            train_set = get_cifar10_dataset("train", aug_level)
            test_set = get_cifar10_dataset("test")  # aug_level 무시됨

            train_loader = DataLoader(
                train_set,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"]
            )
            test_loader = DataLoader(
                test_set,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["num_workers"]
            )

            # 손실 함수 및 옵티마이저
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

            # 학습 루프
            for epoch in range(config["epochs"]):
                print(f"[Epoch {epoch + 1}/{config['epochs']}]")

                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

                print(f"  ✔ Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  ✔ Test  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc
                })

            wandb.finish()
            print("\n")  # 다음 실험 구분


if __name__ == "__main__":
    main()

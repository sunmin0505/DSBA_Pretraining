import wandb
from tqdm import tqdm
from transformers import set_seed
from accelerate import Accelerator

import torch
import omegaconf
from omegaconf import OmegaConf

from utils import load_config
from model import EncoderForClassification
from data import get_dataloader

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs : omegaconf.DictConfig):
    set_seed(int(configs.train_config.seed))
    # Accelerator
    accum_step = int(getattr(configs.train_config, "accum_step", 1))
    accelerator = Accelerator(
        gradient_accumulation_steps=accum_step,
        log_with="wandb"
    )
    device = accelerator.device

    #wandb
    accelerator.init_trackers(
        project_name = configs.train_config.get("project"),
        config = {
            **OmegaConf.to_container(configs, resolve=True),
            "per_device_batch_size": int(getattr(configs.data_config, 'batch_size')),
            "accum_step": accum_step,
            "effective_batch_size": int(getattr(configs.data_config, 'batch_size')) * accum_step
        },
        init_kwargs={"wandb": {"name": configs.train_config.get("run_name")}}
    )

    # Load model
    model = EncoderForClassification(configs.model_config)
    model.train()

    # Set optimizer
    lr = float(configs.train_config.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Load data
    train_loader = get_dataloader(configs.data_config, 'train')
    valid_loader = get_dataloader(configs.data_config, 'valid')
    test_loader = get_dataloader(configs.data_config, 'test')

    # 장치 배치
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )

    # Train & validation for each epoch
    epochs = int(configs.train_config.epochs)

    for epoch in range(1, epochs + 1):
        model.train()

        window_loss_sum = 0.0
        window_acc_sum = 0.0
        window_count = 0
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}"), start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs['loss'] / accelerator.gradient_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
    
            # 로깅
            acc = calculate_accuracy(outputs['logits'], batch['label'])
            bsz = batch['label'].size(0)
            true_loss = float(outputs['loss'].detach().item())

            # 윈도우 평균을 만드기 위해 합계로 유지
            window_loss_sum += true_loss * bsz
            window_acc_sum += acc * bsz
            window_count += bsz

            if accelerator.sync_gradients:
                train_loss = window_loss_sum / max(1, window_count)
                train_acc = window_acc_sum / max(1, window_count)
                accelerator.log({"train_loss": train_loss, "train_accuracy": train_acc, "epoch": epoch})
                # 다음 윈도우를 위해 초기화
                window_loss_sum = 0.0
                window_acc_sum = 0.0
                window_count = 0

        if window_count > 0:
            train_loss = window_loss_sum / max(1, window_count)
            train_acc = window_acc_sum / max(1, window_count)
            accelerator.log({"train_loss": train_loss, "train_accuracy": train_acc, "epoch": epoch})
            window_loss_sum = 0.0
            window_acc_sum = 0.0
            window_count = 0


        # validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid epoch {epoch}"):
                loss, accuracy = valid_iter(model, batch, device)
                bsz = batch['label'].size(0)
                val_loss_sum += float(loss.detach().item()) * bsz
                val_correct += accuracy * bsz
                val_count += bsz

        val_loss = val_loss_sum / max(1, val_count)
        val_accuracy = val_correct / max(1, val_count)
        accelerator.log({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch})

    # test (마지막 모델 사용)
    model.eval()

    total_correct, total_count = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            _, acc = valid_iter(model, batch, device)
            bsz = batch["label"].size(0)
            total_correct += acc * bsz
            total_count += bsz

    test_acc = total_correct / max(1, total_count)
    wandb.log({"test_acc": test_acc})
    print(f"[Last checkpoint] test_acc={test_acc:.4f}")

    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__" :
    configs = load_config('modernbert-1024')
    main(configs)
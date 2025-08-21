import wandb 
from tqdm import tqdm
from transformers import set_seed

import torch
import omegaconf
from omegaconf import OmegaConf

from utils import load_config
from model import EncoderForClassification
from data import get_dataloader
 
def train_iter(model, inputs, optimizer, device, epoch):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({'train_loss' : loss.item(), 'train_acc': accuracy, 'epoch' : epoch})
    
    return loss

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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #wandb
    wandb.init(
        project=configs.train_config.get("project"),
        name=configs.train_config.get("run_name"),
        config=OmegaConf.to_container(configs, resolve=True),
    )

    # Load model
    model = EncoderForClassification(configs.model_config).to(device)
    model.train()

    # Load data
    train_loader = get_dataloader(configs.data_config, 'train')
    valid_loader = get_dataloader(configs.data_config, 'valid')
    test_loader = get_dataloader(configs.data_config, 'test')

    # Set optimizer
    lr = float(configs.train_config.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Train & validation for each epoch
    best_acc = -1.0
    best_path = 'best.pt'

    epochs = int(configs.train_config.epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            train_iter(model, batch, optimizer, device, epoch)
    
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid epoch {epoch}"):
                loss, accuracy = valid_iter(model, batch, device)
                bsz = batch['label'].size(0)
                val_loss_sum += loss.item() * bsz
                val_correct += accuracy * bsz
                val_count += bsz

        val_loss = val_loss_sum / max(1, val_count)
        val_accuracy = val_correct / max(1, val_count)
        wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch})


        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), best_path)

    # test
    model.load_state_dict(torch.load(best_path, map_location=device))
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
    print(f"[Best checkpoint] test_acc={test_acc:.4f}")
    wandb.finish()
    
if __name__ == "__main__" :
    configs = load_config('modernbert')
    main(configs)
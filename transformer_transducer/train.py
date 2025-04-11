import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer
from tqdm import tqdm
from models.loss import RNNTLoss
import argparse
import yaml
import os 

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        speech = batch["fbank"].to(device)
        text = batch["text"].to(device)
        speech_mask = batch["fbank_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        fbank_len = batch["fbank_len"].to(device)
        text_len = batch["text_len"].to(device)

        optimizer.zero_grad()

        output, _, _ = model(
            speech=speech,
            speech_mask=speech_mask,
            text=text,
            text_mask=text_mask,
        )

        # Bỏ <s> ở đầu nếu có
        loss = criterion(output, text, fbank_len, text_len)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # === In loss từng batch ===
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Average training loss: {avg_loss:.4f}")
    return avg_loss


from torchaudio.functional import rnnt_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            speech = batch["fbank"].to(device)
            text = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)

            output, _, _ = model(
                speech=speech,
                speech_mask=speech_mask,
                text=text,
                text_mask=text_mask,
            )

            loss = criterion(output, text, fbank_len, text_len)
            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Average validation loss: {avg_loss:.4f}")
    return avg_loss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def main():
    from torch.optim import Adam
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']
    optimizer_cfg = config['optimizer']


    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path']
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size= training_cfg['batch_size'],
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTransducer(
        in_features=config['model']['in_features'],
        n_classes=len(train_dataset.vocab),
        n_layers=config['model']['n_layers'],
        n_dec_layers=config['model']['n_dec_layers'],
        d_model=config['model']['d_model'],
        ff_size=config['model']['ff_size'],
        h=config['model']['h'],
        joint_size=config['model']['joint_size'],
        enc_left_size=config['model']['enc_left_size'],
        enc_right_size=config['model']['enc_right_size'],
        dec_left_size=config['model']['dec_left_size'],
        dec_right_size=config['model']['dec_right_size'],
        p_dropout=config['model']['p_dropout']
    ).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Khởi tạo loss ===
    # Giả sử <blank> = 0, và bạn chưa dùng reduction 'mean' toàn bộ batch
    criterion = RNNTLoss(config["rnnt_loss"]["blank"] , config["rnnt_loss"]["reduction"])  # hoặc "sum" nếu bạn custom average

    # === Optimizer ===
    optimizer = Adam(model.parameters(), lr=optimizer_cfg['lr'])

    # === Huấn luyện ===
    num_epochs = config["training"]["epochs"]

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model,  dev_loader, criterion, device)

        print(f"📘 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        # Save model checkpoint

        model_filename = os.path.join(
            config['training']['save_path'],
            f"transformer_transducer_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)


if __name__ == "__main__":
    main()

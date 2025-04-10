import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer
from tqdm import tqdm


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

class RNNTLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, logits, targets, fbank_len, text_len):
        # logits: [B, T, U, vocab_size]
        # targets: [B, U]
        # fbank_len: [B]
        # text_len: [B]

        # print(logits.shape)
        # print(targets.shape)
        # print(fbank_len.shape)
        # print(text_len.shape)

        loss = rnnt_loss(logits, targets[:, 1:].int(), fbank_len.int(), text_len.int() - 1, blank=self.blank)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def main():
    from torch.optim import Adam
    
    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path="/home/anhkhoa/transformer_transducer/data/train.json",
        vocab_path="/home/anhkhoa/transformer_transducer/data/vocab.json"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    dev_dataset = Speech2Text(
        json_path="/home/anhkhoa/transformer_transducer/data/dev.json",
        vocab_path="/home/anhkhoa/transformer_transducer/data/vocab.json"
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn = speech_collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerTransducer(
        in_features=80,
        n_classes=len(train_dataset.vocab),
        n_layers=4,
        n_dec_layers=2,
        d_model=256,
        ff_size=1024,
        h=4,
        joint_size=512,
        enc_left_size=2,
        enc_right_size=2,
        dec_left_size=1,
        dec_right_size=1,
        p_dropout=0.1
    ).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Khởi tạo loss ===
    # Giả sử <blank> = 0, và bạn chưa dùng reduction 'mean' toàn bộ batch
    criterion = RNNTLoss(blank = 0 , reduction="mean")  # hoặc "sum" nếu bạn custom average

    # === Optimizer ===
    optimizer = Adam(model.parameters(), lr=1e-3)

    # === Huấn luyện ===
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model,  dev_loader, criterion, device)

        print(f"📘 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()

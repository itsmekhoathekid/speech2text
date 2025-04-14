import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer
from tqdm import tqdm
from models.loss import RNNTLoss
import argparse
import yaml
import os 
import torch.nn.functional as F
from jiwer import wer, cer

ENC_OUT_KEY = "encoder_out"
SPEECH_IDX_KEY = "speech_idx"
HIDDEN_STATE_KEY = "hidden_state"
DECODER_OUT_KEY = "decoder_out"
PREDS_KEY = "preds"
PREV_HIDDEN_STATE_KEY = "prev_hidden_state"

import re

def find_latest_checkpoint(save_path):
    pattern = re.compile(r"transformer_transducer_epoch_(\d+)")
    latest_epoch = -1
    latest_file = None

    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = filename

    if latest_file:
        return os.path.join(save_path, latest_file), latest_epoch
    else:
        return None, 0


class TransducerPredictor:
    """
    Wrapper cho model TransformerTransducer, cài đặt hàm predict dùng cho inference theo kiểu incremental.
    
    Các tham số:
      - model: mô hình đã load checkpoint.
      - vocab: từ điển (dict token -> index) được load từ dataset.
      - device: thiết bị dùng (CPU hoặc GPU).
      - sos, eos, blank: các chỉ số token đặc biệt.
    """
    def __init__(self, model: TransformerTransducer, vocab: dict, device: torch.device,
                 sos: int = 1, eos: int = 2, blank: int = 0):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.sos = sos
        self.eos = eos
        self.blank = blank
        # Tạo mapping index -> token
        self.idx2token = {idx: token for token, idx in vocab.items()}
        
    def model_predict_step(self, speech: torch.Tensor, speech_mask: torch.Tensor, state: dict) -> dict:
        """
        Thực hiện một bước decode:
          - Lấy input text là chuỗi token đã dự đoán (state[PREDS_KEY]).
          - Gọi forward của model với speech, speech_mask, text và text_mask.
          - Từ output (shape [B, M, N, n_classes]), lấy logits tại encoder time step 0 và token cuối của decoder.
          - Tính log softmax, chọn token có xác suất cao nhất (argmax), rồi cập nhật state.
        """
        text = state[PREDS_KEY]
        text_mask = torch.ones_like(text, dtype=torch.bool, device=self.device)
        output, _, _ = self.model(
            speech=speech, 
            speech_mask=speech_mask, 
            text=text, 
            text_mask=text_mask
        )
        logits = output[:, 0, -1, :]  # [B, n_classes]
        logits = F.log_softmax(logits, dim=-1)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
        new_state = state.copy()
        new_state[PREDS_KEY] = torch.cat([state[PREDS_KEY], next_token], dim=1)
        if ENC_OUT_KEY not in state:
            encoder_out, _ = self.model.encoder(speech, speech_mask)
            new_state[ENC_OUT_KEY] = encoder_out
        return new_state
    
    def predict(self, speech: torch.Tensor) -> str:
        """
        Thực hiện decode cho một mẫu (batch size = 1) theo vòng lặp incremental:
          - Khởi tạo state với token start.
          - Lặp cho đến khi nhận được token eos hoặc đạt số bước tối đa.
          - Trả về chuỗi transcription (sau khi chuyển token id sang token string).
        """
        self.model.eval()
        with torch.no_grad():
            B = speech.size(0)
            speech_mask = torch.ones(B, speech.size(1), dtype=torch.bool, device=self.device)
            state = {
                PREDS_KEY: torch.LongTensor([[self.sos]]).to(self.device),
                SPEECH_IDX_KEY: 0,
                HIDDEN_STATE_KEY: None
            }
            max_steps = 100
            for _ in range(max_steps):
                state = self.model_predict_step(speech, speech_mask, state)
                last_pred = state[PREDS_KEY][0, -1].item()
                if last_pred == self.blank:
                    # Có thể cập nhật state nếu cần xử lý trường hợp token blank
                    pass
                if last_pred == self.eos:
                    state[PREDS_KEY] = state[PREDS_KEY][:, :-1]  # Loại bỏ token eos khỏi kết quả
                    break
            pred_ids = state[PREDS_KEY][0, 1:].tolist()  # Loại bỏ token start
            print(pred_ids)
            tokens = [self.idx2token.get(token, "") for token in pred_ids]
            transcription = " ".join(tokens)
            return transcription

def run_inference(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Speech2Text(
        json_path=config['training']['test_path'],
        vocab_path=config['training']['vocab_path']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,      # inference theo mẫu
        shuffle=False,
        collate_fn=speech_collate_fn
    )
    vocab = test_dataset.vocab.stoi  # dạng: {token: index}
    vocab_len = len(vocab)
    # Load model từ checkpoint cuối cùng (dựa theo số epoch được config)
    # model = load_model(config, vocab_len, device)
    predictor = TransducerPredictor(model, vocab, device, sos=1, eos=2, blank=0)
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(test_loader, desc="Inference"):
        speech = batch["fbank"].to(device)
        pred_transcription = predictor.predict(speech)
        all_predictions.append(pred_transcription)
        
        # Ground truth: chuyển từ token id thành chuỗi
        ref_ids = batch["text"].squeeze(0).tolist()
        idx2token = {idx: token for token, idx in vocab.items()}
        ref_tokens = [idx2token.get(token, "") for token in ref_ids]
        ref_transcription = " ".join(ref_tokens)
        all_references.append(ref_transcription)
        print(ref_transcription)
        break
    
    wer_score = wer(all_references, all_predictions)
    cer_score = cer(all_references, all_predictions)
    
    print("\n----- Inference Results -----")
    for i, (ref, pred) in enumerate(zip(all_references, all_predictions)):
        print(f"Sample {i}:")
        print(f"  Reference: {ref}")
        print(f"  Prediction: {pred}\n")
    
    print(f"Average WER: {wer_score:.4f}")
    print(f"Average CER: {cer_score:.4f}")


def train_one_epoch(model, dataloader, optimizer, criterion, device, config):
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

        # run_inference(model, config)
        # logits = output[:, 0, -1, :]
        # print("Logits:", logits)   # kiểm tra giá trị trước khi softmax
        # logits = F.log_softmax(logits, dim=-1)
        # print("Log-softmax logits:", logits)
        

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

    # ==== Load Datasets ====
    train_dataset = Speech2Text(json_path=training_cfg['train_path'], vocab_path=training_cfg['vocab_path'])
    dev_dataset = Speech2Text(json_path=training_cfg['dev_path'], vocab_path=training_cfg['vocab_path'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True, collate_fn=speech_collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=training_cfg['batch_size'], shuffle=False, collate_fn=speech_collate_fn)

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

    criterion = RNNTLoss(config["rnnt_loss"]["blank"], config["rnnt_loss"]["reduction"])
    optimizer = Adam(model.parameters(), lr=optimizer_cfg['lr'])

    # === Load latest checkpoint if exists ===
    save_path = config['training']['save_path']
    latest_ckpt, start_epoch = find_latest_checkpoint(save_path)

    if latest_ckpt:
        print(f"✅ Found checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch += 1
    else:
        print("🆕 No checkpoint found. Starting from scratch.")
        start_epoch = 1

    # === Huấn luyện ===
    num_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss = evaluate(model, dev_loader, criterion, device)

        print(f"📘 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # === Save checkpoint ===
        model_filename = os.path.join(save_path, f"transformer_transducer_epoch_{epoch}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

if __name__ == "__main__":
    main()

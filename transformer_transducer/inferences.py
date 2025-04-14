import torch
import torch.nn.functional as F
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import TransformerTransducer
import yaml
import os
import argparse
from tqdm import tqdm
from jiwer import wer, cer

ENC_OUT_KEY = "encoder_out"
SPEECH_IDX_KEY = "speech_idx"
HIDDEN_STATE_KEY = "hidden_state"
DECODER_OUT_KEY = "decoder_out"
PREDS_KEY = "preds"
PREV_HIDDEN_STATE_KEY = "prev_hidden_state"

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, vocab_len: int, device: torch.device) -> TransformerTransducer:
    checkpoint_path = os.path.join(
        config['training']['save_path'],
        f"transformer_transducer_epoch_1"
    )
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = TransformerTransducer(
        in_features=config['model']['in_features'],
        n_classes=vocab_len,
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

class TransducerPredictor:
    def __init__(self, model, vocab, device, sos=1, eos=2, blank=4):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.sos = sos
        self.eos = eos
        self.blank = blank
        self.idx2token = {idx: token for token, idx in vocab.items()}

    def model_predict_step(self, speech, speech_mask, state):
        text = state[PREDS_KEY]
        text_mask = torch.ones_like(text, dtype=torch.bool, device=self.device)

        if ENC_OUT_KEY not in state:
            encoder_out, _ = self.model.encoder(speech, speech_mask)
            encoder_out = self.model.audio_fc(encoder_out)
            state[ENC_OUT_KEY] = encoder_out
            state[SPEECH_IDX_KEY] = 0
            state["step"] = 0  # add step tracking

        t_idx = state[SPEECH_IDX_KEY]
        encoder_out = state[ENC_OUT_KEY][:, :t_idx + 1, :]

        decoder_out, _ = self.model.decoder(text, text_mask)
        decoder_out = self.model.text_fc(decoder_out)

        output = self.model._join(
            encoder_out[:, -1:, :],
            decoder_out[:, -1:, :]
        )

        logits = output[:, 0, 0, :]
        logits = F.log_softmax(logits, dim=-1)

        # ✅ CHẶN BLANK TRONG BƯỚC ĐẦU TIÊN
        if state["step"] == 0:
            logits[:, self.blank] = -float("inf")

        # print(torch.topk(logits[0], 5))  # debug

        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        new_state = state.copy()
        new_state[PREDS_KEY] = torch.cat([state[PREDS_KEY], next_token], dim=1)
        new_state[SPEECH_IDX_KEY] = t_idx + 1
        new_state["step"] = state["step"] + 1
        return new_state

    def predict(self, speech):
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
            blank_count = 0
            for _ in range(max_steps):
                state = self.model_predict_step(speech, speech_mask, state)
                last_pred = state[PREDS_KEY][0, -1].item()

                if last_pred == self.blank:
                    blank_count += 1
                    if blank_count >= 5:
                        break
                else:
                    blank_count = 0

                if last_pred == self.eos:
                    state[PREDS_KEY] = state[PREDS_KEY][:, :-1]
                    break

            pred_ids = state[PREDS_KEY][0, 1:].tolist()
            tokens = [self.idx2token.get(tid, "") for tid in pred_ids]
            return " ".join(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = Speech2Text(
        json_path=config['training']['test_path'],
        vocab_path=config['training']['vocab_path']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=speech_collate_fn
    )
    vocab = test_dataset.vocab.stoi
    vocab_len = len(vocab)

    model = load_model(config, vocab_len, device)
    predictor = TransducerPredictor(model, vocab, device, sos=1, eos=2, blank=4)

    all_predictions = []
    all_references = []

    for batch in tqdm(test_loader, desc="Inference"):
        speech = batch["fbank"].to(device)
        pred_transcription = predictor.predict(speech)
        all_predictions.append(pred_transcription)

        ref_ids = batch["text"].squeeze(0).tolist()
        idx2token = {idx: token for token, idx in vocab.items()}
        ref_tokens = [idx2token.get(token, "") for token in ref_ids]
        ref_transcription = " ".join(ref_tokens)
        print("🔊", pred_transcription)
        print("🎯", ref_transcription)
        all_references.append(ref_transcription)

    wer_score = wer(all_references, all_predictions)
    cer_score = cer(all_references, all_predictions)

    print("\n----- Inference Results -----")
    for i, (ref, pred) in enumerate(zip(all_references, all_predictions)):
        print(f"Sample {i}:")
        print(f"  Reference: {ref}")
        print(f"  Prediction: {pred}")
        print()

    print(f"Average WER: {wer_score:.4f}")
    print(f"Average CER: {cer_score:.4f}")

if __name__ == "__main__":
    main()

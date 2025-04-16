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
        f"transformer_transducer_epoch_19"
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

    def beam_search(self, speech, beam_width=5):
        self.model.eval()
        with torch.no_grad():
            speech_mask = torch.ones(speech.size(0), speech.size(1), dtype=torch.bool, device=self.device)
            encoder_out, _ = self.model.encoder(speech, speech_mask)  # (1, T, D)

            beam = [{
                "tokens": [self.sos],
                "log_prob": 0.0,
                "state": None
            }]
            completed_hypotheses = []
            T = encoder_out.size(1)

            for t in range(T):
                new_beam = []
                for hyp in beam:
                    prev_tokens = torch.LongTensor(hyp["tokens"]).unsqueeze(0).to(self.device)
                    decoder_mask = torch.ones_like(prev_tokens, dtype=torch.bool, device=self.device)
                    decoder_out, _ = self.model.decoder(prev_tokens, decoder_mask)
                    decoder_out = decoder_out[:, -1:, :]  # (1, 1, D)

                    enc_out_t = encoder_out[:, t:t+1, :]
                    enc_out_t = self.model.audio_fc(enc_out_t)
                    dec_out_t = self.model.text_fc(decoder_out)
                    joint_out = self.model._join(enc_out_t, dec_out_t)
                    log_probs = F.log_softmax(joint_out.squeeze(1), dim=-1).squeeze(0)  # (vocab_size,) hoặc (1, vocab_size)
                    log_probs = log_probs.view(-1)  # đảm bảo đúng shape (vocab_size,)

                    topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
                    topk_log_probs = topk_log_probs.tolist()
                    topk_ids = topk_ids.tolist()

                    for i in range(beam_width):
                        new_token = topk_ids[i]
                        new_log_prob = hyp["log_prob"] + topk_log_probs[i]
                        new_tokens = hyp["tokens"] + [new_token]

                        if new_token == self.eos:
                            completed_hypotheses.append({
                                "tokens": new_tokens[1:-1],
                                "log_prob": new_log_prob
                            })
                        elif new_token == self.blank:
                            new_beam.append({
                                "tokens": hyp["tokens"],
                                "log_prob": new_log_prob,
                                "state": hyp["state"]
                            })
                        else:
                            new_beam.append({
                                "tokens": new_tokens,
                                "log_prob": new_log_prob,
                                "state": hyp["state"]
                            })

                beam = sorted(new_beam, key=lambda x: x["log_prob"], reverse=True)[:beam_width]

            if not completed_hypotheses:
                completed_hypotheses = beam

            best_hyp = max(completed_hypotheses, key=lambda x: x["log_prob"])
            tokens = [self.idx2token.get(t, "") for t in best_hyp["tokens"]]
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
        pred_transcription = predictor.beam_search(speech, beam_width=5)
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

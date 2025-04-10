import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T

# [{idx : {encoded_text : Tensor, wav_path : text} }]


def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    import json

    with open(path, "r") as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def __len__(self):
        return len(self.vocab)

class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()

    def __len__(self):
        return len(self.data)
    
    

    def get_fbank(self, wav_path):
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.squeeze(0)

        pre_emphasis = 0.97
        waveform = torch.cat((waveform[:1], waveform[1:] - pre_emphasis * waveform[:-1]))

        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        log_mel_spec = T.AmplitudeToDB(stype='power', top_db=80)(mel_spec)

        # CMVN per sample (utterance-level)
        mean = log_mel_spec.mean(dim=1, keepdim=True)
        std = log_mel_spec.std(dim=1, keepdim=True)
        normalized_log_mel_spec = (log_mel_spec - mean) / (std + 1e-5)

        return normalized_log_mel_spec

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        fbank = self.get_fbank(wav_path).transpose(0, 1).float()  # [T, 80]

        return {
            "text": encoded_text,        # [T_text]
            "fbank": fbank,              # [T_audio, 80]
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0]
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def speech_collate_fn(batch):
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)

    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 80]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    text_mask=calculate_mask(text_lens, padded_texts.size(1))

    return {
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask
    }
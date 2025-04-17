import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from warp_rnnt import rnnt_loss
import scipy
from model_rnnt.eval_distance import eval_wer, eval_cer
from model_rnnt.model import Transducer
from model_rnnt.encoder import BaseEncoder
from model_rnnt.decoder import BaseDecoder
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict
from tensorboardX import SummaryWriter
import k2

# class TransducerLoss(nn.Module):
#     def __init__(self, blank_id=0, reduction='mean', gather=True):
#         super().__init__()
#         self.blank_id = blank_id
#         self.reduction = reduction
#         self.gather = gather

#     def forward(self, logits, targets, input_lengths, target_lengths):
#         logits = torch.nn.functional.log_softmax(logits, dim=-1)  # Áp dụng log-softmax
#         return rnnt_loss(
#             logits, targets, input_lengths, target_lengths,
#             reduction=self.reduction, blank=self.blank_id, gather=self.gather
#         )
        


def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue
            
            index, char = line.split(',')

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

def computer_cer(preds, labels):
    char2index, index2char = load_label('/data/npl/Speech2Text/RNN-Transducer/data/dic.labels')
    
    total_wer = 0
    total_cer = 0

    total_wer_len = 0
    total_cer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            units.append(index2char[a])
            
        for b in pred:
            units_pred.append(index2char[b])

        label = moasseugi(units)
        pred = moasseugi(units_pred)
    
        wer = eval_wer(pred, label)
        cer = eval_cer(pred, label)
        
        wer_len = len(label.split())
        cer_len = len(label.replace(" ", ""))

        total_wer += wer
        total_cer += cer

        total_wer_len += wer_len
        total_cer_len += cer_len

    return total_wer, total_cer, total_wer_len, total_cer_len

def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()
    total_batch_num = len(train_loader)

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets, inputs_lengths, targets_lengths = data
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_lengths, targets_lengths = inputs_lengths.to(device), targets_lengths.to(device)

        logits, loss = model(inputs, inputs_lengths, targets, targets_lengths)
        total_loss += loss.item()
        print(5)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"[{datetime.datetime.now()}] Train Batch {i}/{total_batch_num} - Loss: {loss.item():.4f}")
    
    train_loss = total_loss / total_batch_num
    train_time = time.time() - start_time
    print(f"✅ Epoch Training Loss: {train_loss:.4f}, Time: {train_time:.2f}s")
    return train_loss

def eval(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_batch_num = len(val_loader)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, inputs_lengths, targets_lengths = data

            inputs, targets = inputs.to(device), targets.to(device)
            inputs_lengths, targets_lengths = inputs_lengths.to(device), targets_lengths.to(device)

            logits, loss = model(inputs, inputs_lengths, targets, targets_lengths)

            total_loss += loss.item()

    val_loss = total_loss / total_batch_num
    print(f"✅ Epoch Validation Loss: {val_loss:.4f}")
    return val_loss
from scipy.signal import windows as sp_windows

windows = {
    'hamming': sp_windows.hamming,
    'hann': sp_windows.hann,
    'blackman': sp_windows.blackman,
    'bartlett': sp_windows.bartlett
}

def main():
    yaml_name = "/data/npl/Speech2Text/RNN-Transducer/label,csv/RNN-T_mobile_2.yaml"
    with open("./train.txt", "w") as f:
        f.write(yaml_name)
        f.write('\n')
        f.write('\n')
        f.write("학습 시작")
        f.write('\n')

    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    
    summary = SummaryWriter()



    SAMPLE_RATE = config.audio_data.sampling_rate
    WINDOW_SIZE = config.audio_data.window_size
    WINDOW_STRIDE = config.audio_data.window_stride
    WINDOW = config.audio_data.window

    audio_conf = dict(sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE,
                        window_stride=WINDOW_STRIDE,
                        window=WINDOW)

    train_manifest_filepath = config.data.train_path
    val_manifest_filepath = config.data.val_path
    
    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    #model
    #Prediction Network
    enc = BaseEncoder(input_size=config.model.enc.input_size,
                      hidden_size=config.model.enc.hidden_size, 
                      output_size=config.model.enc.output_size,
                      n_layers=config.model.enc.n_layers, 
                      dropout=config.model.dropout, 
                      bidirectional=config.model.enc.bidirectional)
    
    #Transcription Network
    dec = BaseDecoder(embedding_size=config.model.dec.embedding_size,
                      hidden_size=config.model.dec.hidden_size, 
                      vocab_size=config.model.vocab_size, 
                      output_size=config.model.dec.output_size, 
                      n_layers=config.model.dec.n_layers, 
                      dropout=config.model.dropout)

    model = Transducer(enc, dec, config.model.joint.input_size, config.model.joint.inner_dim, config.model.vocab_size).to(device)
    print("🛠 Model Architecture:")
    print(model)
    print("\n🔍 Model Parameters:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.optim.milestones, gamma=config.optim.decay_rate)
    criterion = None # TransducerLoss(blank_id=config.loss.blank_id).to(device)
    scaler = None

    train_dataset = SpectrogramDataset(audio_conf, 
                                       config.data.train_path,
                                       feature_type=config.audio_data.type, 
                                       normalize=True, 
                                       spec_augment=True)

    train_loader = AudioDataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)
    
    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)
    
    print("🚀 Training started...")
    pre_val_loss = float("inf")
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        print(f"\n📌 {datetime.datetime.now()} - Epoch {epoch+1} Training Start")
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        print(f"\n📌 {datetime.datetime.now()} - Epoch {epoch+1} Validation Start")
        val_loss = eval(model, val_loader, criterion, device)
        scheduler.step()
        if val_loss < pre_val_loss:
            print("✅ Saving best model...")
            torch.save(model.state_dict(), f"/data/npl/Speech2Text/RNN-Transducer/checkpoint/epoch_{epoch+1}.pth")
            pre_val_loss = val_loss

if __name__ == "__main__":
    main()

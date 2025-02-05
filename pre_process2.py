
import os
import pickle

from tqdm import tqdm

from config import wav_folder, pickle_file
from utils import ensure_folder, parse_args
import json
import re

def normalize_script(script: str) -> str:
    # Thay thế số bằng chữ
    script = script.lower()
    script = script.replace("0", " không ")
    script = script.replace("1", " một ")
    script = script.replace("2", " hai ")
    script = script.replace("3", " ba ")
    script = script.replace("4", " bốn ")
    script = script.replace("5", " năm ")
    script = script.replace("6", " sáu ")
    script = script.replace("7", " bảy ")
    script = script.replace("8", " tám ")
    script = script.replace("9", " chín ")

    # Regex chỉ thay thế ký tự đặc biệt, KHÔNG thay đổi chữ cái
    script = re.sub(r"[^\w\sÀ-ỹ]", " ", script)
    
    # Xóa khoảng trắng thừa
    script = "".join(script.split())
    
    return script


def get_data(split, n_samples, tran_file):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(tran_file, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    # tran_dict = dict()
    # for id, line in lines.items():
    #     trn = normalize_script(line['script'])
    #     key = id
    #     tran_dict[key] = trn

    samples = []

    #n_samples = 5000
    # rest = n_samples    
    
    folder = wav_folder
    ensure_folder(folder)
    # dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.exists(os.path.join(folder, d))]
    
    for idx, data in datas.items():
        wave = os.path.join(folder, data['voice'].split('.')[0] + '.wav')
        trn = normalize_script(data['script'])
        # key = idx

        trn = list(trn.strip()) + ['<eos>']
        for token in trn:
            build_vocab(token)
        trn = [VOCAB[token] for token in trn]
        samples.append({'trn': trn, 'wave': wave})

    samples = samples[:n_samples] if n_samples > 0 else samples

    # for dir in tqdm(dirs):
    #     files = [f for f in os.listdir(dir) if f.endswith('.wav')]

    #     rest = len(files) if n_samples <= 0 else rest

    #     for f in files[:rest]:
            
    #         wave = os.path.join(dir, f)
    #         key = f.split('.')[0]

    #         if key in tran_dict:
    #             trn = tran_dict[key]
    #             trn = list(trn.strip()) + ['<eos>']

    #             for token in trn:
    #                 build_vocab(token)

    #             trn = [VOCAB[token] for token in trn]

    #             samples.append({'trn': trn, 'wave': wave})
        
    #     rest = rest - len(files) if n_samples > 0 else rest
    #     if rest <= 0 :
    #         break  

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples

def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":

    # number of examples to use 
    global args
    args = parse_args()
    tmp = args.n_samples.split(",")
    tmp = [a.split(":") for a in tmp]
    tmp = {a[0]:int(a[1]) for a in tmp}
    args.n_samples = {"train":-1, "dev":-1,"test":-1}
    args.n_samples.update(tmp)
    
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}

    paths = {
        'train' : r"C:\Users\VIET HOANG - VTS\Desktop\data\train.json",
        'dev' : r"C:\Users\VIET HOANG - VTS\Desktop\data\dev.json",
        'test' : r"C:\Users\VIET HOANG - VTS\Desktop\data\test.json"
    }

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train', args.n_samples["train"], paths['train'])
    data['dev'] = get_data('dev', args.n_samples["dev"], paths['dev'])
    data['test'] = get_data('test', args.n_samples["test"], paths['test'])
    
    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
    
    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))
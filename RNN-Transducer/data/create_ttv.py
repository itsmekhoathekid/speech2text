import json
import os
import csv

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None

def save_csv(data, save_path):
    """
    Lưu dữ liệu thành file CSV.
    """
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(save_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "text"])  # Tiêu đề cột
        for wav_path, text in data:
            writer.writerow([wav_path, text])
    
    print(f"File CSV đã được lưu tại: {save_path}")

def encode(stri, dic):
    return " ".join([str(dic.get(word)) for word in stri.strip()])

def process_data(json_path, save_csv_path, dic_path):
    data = load_json(json_path)
    if not data:
        return
    dic = load_json(dic_path)
    processed_data = [(info["wav"], encode(info["text"], dic)) for info in data.values()]
    save_csv(processed_data, save_csv_path)


train_path = '/data/npl/Speech2Text/Speech-Transformer-master/vlsp_data/custom_data/train_vlsp.json'
test_path = '/data/npl/Speech2Text/Speech-Transformer-master/vlsp_data/custom_data/test_vlsp.json'
output_train = '/data/npl/Speech2Text/RNN-Transducer/data/train.csv'
output_test = '/data/npl/Speech2Text/RNN-Transducer/data/test.csv'
dic_path = '/data/npl/Speech2Text/Speech-Transformer/data/dic.json'

process_data(train_path, output_train, dic_path)
process_data(test_path, output_test, dic_path)
